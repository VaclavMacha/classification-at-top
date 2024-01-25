from typing import Iterable, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Iterable]
Device = Union[str, torch.device]


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
