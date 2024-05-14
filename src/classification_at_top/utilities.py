import torch
from torch import Tensor


def find_extrema(x: Tensor, reverse: bool = True) -> Tensor:
    """
    Finds the index of the largest/smallest value in the given tensor.

    Args:
        x (torch.Tensor): The input tensor.
        reverse (bool): If True, finds the largest value. If False, finds the smallest value. Defaults to True.

    Returns:
        Tensor: The index of the largest/smallest value in the given tensor.
    """
    if reverse:
        return x.argmax()
    else:
        return x.argmin()


def find_kth(x: Tensor, k: int, reverse: bool = True) -> Tensor:
    """
    Finds the index of the k-th largest/smallest value in the given tensor.

    Args:
        x (torch.Tensor): The input tensor.
        k (int): The index of the element to find.
        reverse (bool): If True, finds the k-th largest value. If False, finds the k-th smallest value. Defaults to True.

    Returns:
        Tensor: The index of the k-th largest/smallest value in the given tensor.

    Raises:
        ValueError: If k is out of range.
    """

    if k == 0:
        return find_extrema(x, reverse)
    elif k > 0 and k < len(x):
        return x.argsort(descending=reverse)[k]
    else:
        raise ValueError(f"Invalid k. Expected 0 <= k < {len(x)}, but got {k}")


def find_quantile(x: Tensor, tau: float, reverse: bool = True) -> Tensor:
    """
    Finds the index of the top/bottom tau-quantile value in the given tensor.

    Args:
        x (torch.Tensor): The input tensor.
        k (int): The index of the element to find.
        reverse (bool): If True, finds the top tau-quantile. If False, the bottom tau-quantile. Defaults to True.

    Returns:
        Tensor: The index of the top/bottom tau-quantile value in the given tensor.

    Raises:
        ValueError: If tau is out of range.
    """
    if not 0 < tau < 1:
        raise ValueError(f"Invalid quantile. Expected 0 < tau < 1, but got {tau=}")

    t = x.quantile(1 - tau if reverse else tau)
    return (torch.abs(x - t)).argmin()
