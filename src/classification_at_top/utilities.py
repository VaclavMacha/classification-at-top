from typing import Iterable, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Iterable]
Device = Union[str, torch.device]
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


def to_torch(x: ArrayLike, device: Device | None = None) -> torch.Tensor:
    """
    Converts an array-like object to a torch.Tensor and pushes it to the given device.

    Args:
        x (ArrayLike): The input array-like object.
        device (Device): The device to push the tensor to. Defaults to None

    Returns:
        torch.Tensor: The resulting tensor located on the given device.

    Raises:
        ValueError: If the input type is not an array-like object.
    """
    if not isinstance(x, ArrayLike):
        raise ValueError(f"Invalid input type. Expected {ArrayLike}")

    return torch.tensor(x).to(device)


def find_kth(x: np.ndarray, k: int, reverse: bool = False) -> Tuple[Number, int]:
    """
    Finds the k-th smallest or largest element in an array.

    Args:
        x (np.ndarray): The input array.
        k (int): The index of the element to find.
        reverse (bool): If True, finds the k-th largest element. If False, finds the k-th smallest element. Defaults to False.

    Returns:
        Tuple[Number, int]: A tuple containing the k-th element and its index.

    Raises:
        ValueError: If k is out of range.
    """
    if k < 0 or k >= x.size:
        raise ValueError(f"Invalid k. Expected 0 <= k < {x.size}, but got {k}")

    if reverse:
        ind = np.argpartition(-x, k)[k]
    else:
        ind = np.argpartition(x, k)[k]
    return x[ind], ind

