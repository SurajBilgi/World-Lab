import argparse
import json
import pathlib

import torch
import torchvision
from jaxtyping import Float
from torch import Tensor
from torchvision.transforms.v2.functional import resize_image

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=pathlib.Path,
    default="models/small-256",
)
parser.add_argument(
    "--input-image",
    type=pathlib.Path,
    default="/images/img1.jpg",
)
parser.add_argument(
    "--output-image",
    type=pathlib.Path,
    default="out.png",
)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")


def read_image(
    p: pathlib.Path,
    size: int,
) -> Float[Tensor, "3 h w"]:
    """Read an image off disk, resize it, and return it in [-1, 1] range"""
    x = torchvision.io.decode_image(str(p))
    x = resize_image(x, [size, size])
    x = x.float() / 127.5 - 1.0
    assert x.shape == (3, size, size)
    return x


def main(args):
    device = torch.device(args.device)
    config_path = args.model / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    model = Model(**config["model_kwargs"]).to(device)
    model_path = args.model / "model.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    x = read_image(args.input_image, config["image_size"]).cuda()
    y = model(x[None])[0]

    # Convert from [-1, 1] back to [0, 255]
    y = ((y + 1.0) * 127.5).round().clamp(min=0, max=255).byte().cpu()
    torchvision.io.write_png(y, args.output_image)


if __name__ == "__main__":
    main(parser.parse_args())
