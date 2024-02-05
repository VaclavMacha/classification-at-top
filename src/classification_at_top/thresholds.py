from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .utilities import (
    Number,
    find_extrema,
    find_kth,
    find_quantile,
    to_numpy,
)


def find_threshold(
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
    if len(y) != len(s):
        raise ValueError(
            f"Expected y and s to have the same length, but got {len(y)} and {len(s)}"
        )

    if not (len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)):
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


class FindThreshold(Function):
    @staticmethod
    def forward(
        y: Tensor,
        s: Tensor,
        by: int | None,
        k_or_tau: int | float | None,
        reverse: bool,
    ):

        t, t_ind = find_threshold(
            to_numpy(y),
            to_numpy(s),
            by=by,
            k_or_tau=k_or_tau,
            reverse=reverse,
        )
        t_tensor = torch.tensor(
            t, device=s.device, dtype=s.dtype, requires_grad=s.requires_grad
        )
        t_ind_tensor = torch.tensor(
            t_ind, device=s.device, dtype=int, requires_grad=False
        )

        return t_tensor, t_ind_tensor

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, s, *_ = inputs
        _, t_ind = output
        ctx.save_for_backward(s, t_ind)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_t, grad_output_t_ind):
        grad_s = None

        if ctx.needs_input_grad[1]:
            s, t_ind = ctx.saved_tensors

            grad_t = torch.zeros_like(s)
            grad_t[t_ind] = 1
            grad_s = grad_output_t * grad_t

        return None, grad_s, None, None, None
