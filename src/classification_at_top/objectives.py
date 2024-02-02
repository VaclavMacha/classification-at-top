import torch
from torch import Tensor

from .utilities import Number


def hinge(x: Tensor, scale: Number) -> Tensor:
    return torch.maximum(torch.zeros_like(x), 1 + scale * x)


def quadratic_hinge(x: Tensor, scale: Number) -> Tensor:
    return torch.maximum(torch.zeros_like(x), 1 + scale * x) ** 2
