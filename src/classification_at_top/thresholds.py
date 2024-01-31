from typing import Tuple

import numpy as np
from attrs import define, field

from .utilities import (
    ArrayLike,
    Number,
    find_extrema,
    find_kth,
    find_quantile,
    to_numpy,
)


def _find_threshold(
    y: np.ndarray,
    s: np.ndarray,
    by: int | None,
    k_or_tau: int | float | None,
    reverse: bool,
) -> Tuple[Number, int]:
    """
    Finds the threshold value and its corresponding index based on the given inputs.

    Args:
        y (np.ndarray): The input array of labels.
        s (np.ndarray): The input array of scores.
        by (int | None): The label value to filter by. If None, all labels are considered.
        k_or_tau (int | float | None): The parameter to determine which function to use to
            find the threshold value:
            If None, the `find_extrema` function is used.
            If int, `find_kth` function is used.
            If float, `find_quantile` function is used.
        reverse (bool, optional): Determines whether to find the maximum or minimum value.

    Returns:
        Tuple[Number, int]: A tuple containing the threshold value and its corresponding index.

    Raises:
        ValueError: If y and s do not have the same shape.
        ValueError: If y and s are not 1-dimensional.
    """
    if y.shape != s.shape:
        raise ValueError(
            f"Expected y and s to have the same shape, but got {y.shape} and {s.shape}"
        )

    if len(y.shape) != 1:
        raise ValueError(f"Expected y and s to be 1-dimensional, but got {y.shape}")

    if isinstance(by, int):
        inds = np.where(y == by)[0]
        s_ = s[inds]
    else:
        inds = np.arange(y.size).reshape(y.shape)
        s_ = s

    if k_or_tau is None:
        t, t_ind = find_extrema(s_, reverse)
    elif isinstance(k_or_tau, int):
        t, t_ind = find_kth(s_, k_or_tau, reverse)
    elif isinstance(k_or_tau, float):
        t, t_ind = find_quantile(s_, k_or_tau, reverse)
    return t, inds[t_ind]


def find_threshold(y: ArrayLike, s: ArrayLike, **kwargs):
    """
    Finds the threshold value and its corresponding index based on the given inputs.

    Args:
        y (ArrayLike): The input array of labels.
        s (ArrayLike): The input array of scores.
        **kwargs: Keyword arguments to be passed to the `_find_threshold` function.

    Returns:
        Tuple[Number, int] or Tuple[np.ndarray, np.ndarray]: A tuple containing the
            threshold value(s) and its corresponding index(es).
            If y is 1-dimensional, the return type is Tuple[Number, int].
            If y is 2-dimensional, the return type is Tuple[np.ndarray, np.ndarray].

    Raises:
        ValueError: If y and s do not have the same shape.
        ValueError: If y and s are not 1-dimensional or 2-dimensional.
    """
    if y.shape != s.shape:
        raise ValueError(
            f"Expected y and s to have the same shape, but got {y.shape} and {s.shape}"
        )

    if len(y.shape) not in [1, 2]:
        raise ValueError(
            f"Expected y and s to be 1- or 2-dimensional, but got {y.shape}"
        )

    y_ = to_numpy(y)
    s_ = to_numpy(s)

    if len(y.shape) == 1:
        t, t_ind = _find_threshold(y_, s_, **kwargs)
    else:
        t = np.zeros((1, s.shape[-1]), dtype=s_.dtype)
        t_ind = np.zeros((1, s.shape[-1]), dtype=int)

        for i in range(y.shape[-1]):
            t[0, i], t_ind[0, i] = _find_threshold(y_[..., i], s_[..., i], **kwargs)

    return t, t_ind

