import pathlib

import torch


class Logger:
    def __init__(self, name: str):
        self.path = pathlib.Path(__file__).parent / f"{name}.log"
        self.initialized = False

    def log(self, msg: str):
        mode = "a" if self.initialized else "w"
        with self.path.open(mode=mode) as f:
            f.write(f"{msg}\n")
        self.initialized = True


logger = Logger(name="activation_stats")


def log_stats(f):
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        if torch.is_tensor(out) and out.ndim == 3:
            means = out.mean(dim=(0, 1)).tolist()
            stds = out.std(dim=(0, 1)).tolist()
        msgs = []
        for i, (mean, std) in enumerate(zip(means, stds)):
            msgs.append(f"dim {i}: {mean} ± {std}")
        msg = "\n".join(msgs)
        msg = f"Layer stats:\n{'-' * 40}\n{msg}\n"
        logger.log(msg)
        return out

    return wrapped
