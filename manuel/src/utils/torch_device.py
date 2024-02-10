import torch


def device() -> str:
    """Returns cuda if available, mps if available, otherwise cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.__version__ >= "2.2.0" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
