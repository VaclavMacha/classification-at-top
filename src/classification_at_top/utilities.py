from typing import Iterable, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Iterable]
Number = Union[int, float]


def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Converts an array-like object to a numpy array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        np.ndarray: The resulting numpy array.

    Raises:
        ValueError: If the input type is not an array-like object.
    """
    if not isinstance(x, ArrayLike):
        raise ValueError(f"Invalid input type. Expected {ArrayLike}")

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


def find_extrema(x: np.ndarray, reverse: bool = False) -> Tuple[Number, int]:
    """
    Finds the minimum/maximum value and its index in the given array.

    Args:
        x (np.ndarray): The input array.
        reverse (bool): If True, finds the largest element. If False, finds the smallest element. Defaults to False.

    Returns:
        Tuple[Number, int]: A tuple containing the maximum value and its index.
    """
    if reverse:
        ind = np.argmax(x)
    else:
        ind = np.argmin(x)

    return x[ind], ind


def find_kth(x: np.ndarray, k: int, reverse: bool = False) -> Tuple[Number, int]:
    """
    Finds the k-th smallest or largest element and its index in the given array.

    Args:
        x (np.ndarray): The input array.
        k (int): The index of the element to find.
        reverse (bool): If True, finds the k-th largest element. If False, finds the k-th
            smallest element. Defaults to False.

    Returns:
        Tuple[Number, int]: A tuple containing the k-th element and its index.

    Raises:
        ValueError: If k is out of range.
    """
    if k < 0 or k > x.size:
        raise ValueError(f"Invalid k. Expected 0 <= k < {x.size}, but got {k}")

    if k == 0:
        return find_extrema(x, reverse)

    if reverse:
        ind = np.argpartition(-x, k)[k]
    else:
        ind = np.argpartition(x, k)[k]
    return x[ind], ind


def find_quantile(
    x: np.ndarray, tau: float, reverse: bool = False
) -> Tuple[Number, int]:
    """
    Finds the tau-quantile and its index of the given array.

    Args:
        x (np.ndarray): The input array.
        tau (float): The quantile value between 0 and 1.
        reverse (bool): If True, finds the quantile from the top of the array. If False,
            finds the quantile from the bottom. Defaults to False.

    Returns:
        Tuple[Number, int]: A tuple containing the quantile value and its index.

    Raises:
        ValueError: If tau is out of range.
    """
    if 0 > tau or tau > 1:
        raise ValueError(f"Invalid quantile. Expected 0 <= tau <= 1, but got {tau}")

    t = np.quantile(x, 1 - tau if reverse else tau)
    ind = (np.abs(x - t)).argmin()
    return x[ind], ind
