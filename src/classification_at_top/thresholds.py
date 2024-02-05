from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .utilities import find_extrema, find_kth, find_quantile


class FindThreshold(Function):
    @staticmethod
    def forward(
        y: Tensor,
        s: Tensor,
        by: int | None,
        k_or_tau: int | float | None,
        reverse: bool,
    ):

        if not y.dim() == s.dim() == 1:
            raise ValueError(
                f"Expected y and s to be 1-dimensional: {y.shape=}, {s.shape=}"
            )

        if len(y) != len(s):
            raise ValueError(
                f"Expected y and s to have the same length: {len(y)=} != {len(s)=}"
            )

        if isinstance(by, int):
            inds = torch.where(y == by)[0]
            s_ = s[inds]
        else:
            inds = None
            s_ = s

        if k_or_tau is None:
            ind = find_extrema(s_, reverse)
        elif isinstance(k_or_tau, int):
            ind = find_kth(s_, k_or_tau, reverse)
        elif isinstance(k_or_tau, float):
            ind = find_quantile(s_, k_or_tau, reverse)

        ind = inds[ind] if isinstance(by, int) else ind
        return s[ind], ind

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, s, *_ = inputs
        _, t_ind = output
        ctx.save_for_backward(s, t_ind)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_t, _):
        grad_s = None

        if ctx.needs_input_grad[1]:
            s, t_ind = ctx.saved_tensors

            grad_t = torch.zeros_like(s)
            grad_t[t_ind] = 1
            grad_s = grad_output_t * grad_t

        return None, grad_s, None, None, None


def find_threshold(
    y: Tensor,
    s: Tensor,
    by: int | None = None,
    k_or_tau: int | float | None = None,
    reverse: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Finds the threshold value and its corresponding index based on the given inputs.

    Args:
        y (Tensor): The input tensor with labels.
        s (Tensor): The input tensor with scores.
        by (int | None): The label value to filter by. If None, all labels are considered.
            Defaults to None.
        k_or_tau (int | float | None): The parameter to determine which function to use to
            find the threshold value:
            If None, the `find_extrema` function is used.
            If int, `find_kth` function is used.
            If float, `find_quantile` function is used.
            Defaults to None.
        reverse (bool, optional): Determines whether to find the maximum or minimum value.
            Defaults to True.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the threshold value and its corresponding
        index.

    Raises:
        ValueError: If y and s do not have the same length.
        ValueError: If y and s are not 1-dimensional.
    """
    return FindThreshold.apply(y, s, by, k_or_tau, reverse)
