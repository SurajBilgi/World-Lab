import math

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

import utils


def multihead_attention(
    q: Float[Tensor, "b h nq dh"],
    k: Float[Tensor, "b h nk dh"],
    v: Float[Tensor, "b h nk dh"],
) -> Float[Tensor, "b h s dh"]:
    dh = q.shape[-1]
    return (q @ k.transpose(2, 3) / math.sqrt(dh)).softmax(dim=3) @ v


class LocalRmsNorm(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int = 7, eps: float = 1e-7):
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be positive and odd; got {kernel_size}")
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(model_dim))

    def forward(self, x: Float[Tensor, "b s*s d"]) -> Float[Tensor, "b s*s d"]:
        b, ss, d = x.shape
        s = int(math.sqrt(ss))
        if s * s != ss:
            raise ValueError("Input must be square; got shape {x.shape}")

        y = einops.rearrange(x, "b (h w) d -> (b d) 1 h w", h=s, w=s)
        y = F.unfold(
            y, kernel_size=self.kernel_size, padding=self.padding
        )  # (b*d, k*k, f)
        rms = torch.sqrt(self.eps + (y * y).mean(dim=1, keepdim=True))  # (b * d, 1, f)
        rms = einops.rearrange(rms, "(b d) 1 (h w) -> b (h w) d", b=b, d=d, h=s, w=s)

        out = x / rms
        out = out * self.weight
        return out


class FFN(nn.Module):
    def __init__(self, *, model_dim: int, ffn_dim: int):
        super().__init__()
        self.ffn1 = nn.Linear(model_dim, ffn_dim)
        self.ffn2 = nn.Linear(model_dim, ffn_dim)
        self.ffn3 = nn.Linear(ffn_dim, model_dim)
        self.norm = LocalRmsNorm(model_dim)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        x = self.norm(x)
        h1 = self.ffn1(x)
        h2 = F.silu(self.ffn2(x))
        out = self.ffn3(h1 * h2)
        return out


class SelfAttention(nn.Module):
    def __init__(self, *, model_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.norm = LocalRmsNorm(model_dim)

    @utils.log_stats
    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        x = self.norm(x)
        q = einops.rearrange(self.q_proj(x), "b s (h dh) -> b h s dh", dh=self.head_dim)
        k = einops.rearrange(self.k_proj(x), "b s (h dh) -> b h s dh", dh=self.head_dim)
        v = einops.rearrange(self.v_proj(x), "b s (h dh) -> b h s dh", dh=self.head_dim)

        attn = multihead_attention(q, k, v)
        attn = einops.rearrange(attn, "b h s dh -> b s (h dh)")
        out = self.out_proj(attn)
        return out


class ParallelBlock(nn.Module):
    def __init__(self, *, model_dim: int, head_dim: int, ffn_dim: int):
        super().__init__()
        self.attn = SelfAttention(model_dim=model_dim, head_dim=head_dim)
        self.ffn = FFN(model_dim=model_dim, ffn_dim=ffn_dim)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return x + self.attn(x) + self.ffn(x)


class Model(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        num_blocks: int,
        model_dim: int,
        head_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        conv_kwargs = {"kernel_size": patch_size, "stride": patch_size}
        self.patchify = nn.Conv2d(3, model_dim, **conv_kwargs)
        self.out_linear = nn.Linear(model_dim, model_dim)
        self.unpatchify = nn.ConvTranspose2d(model_dim, 3, **conv_kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                ParallelBlock(model_dim=model_dim, head_dim=head_dim, ffn_dim=ffn_dim)
            )

    def forward(self, x: Float[Tensor, "b 3 h w"]) -> Float[Tensor, "b 3 h w"]:
        x = self.patchify(x)
        ph, pw = x.shape[2], x.shape[3]
        x = einops.rearrange(x, "b d ph pw -> b (ph pw) d")
        for block in self.blocks:
            x = block(x)
        x = self.out_linear(x).relu()
        x = einops.rearrange(x, "b (ph pw) d -> b d ph pw", ph=ph, pw=pw)
        x = self.unpatchify(x)
        return x
