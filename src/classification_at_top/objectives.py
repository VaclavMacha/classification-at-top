from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from .thresholds import find_threshold

Number = Union[int, float]


def hinge(x: Tensor, scale: Number = 1) -> Tensor:
    return torch.maximum(torch.zeros_like(x), 1 + scale * x)


def quadratic_hinge(x: Tensor, scale: Number = 1) -> Tensor:
    return torch.maximum(torch.zeros_like(x), 1 + scale * x) ** 2


SURROGATES = {
    "hinge": hinge,
    "quadratic_hinge": quadratic_hinge,
}


class ClassificationAtTopObjective(Module):
    def __init__(
        self,
        alpha: float,
        by: int | None,
        k_or_tau: Number | None,
        reverse: bool,
        surrogate: str,
    ):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

        if by is not None and by not in [0, 1]:
            raise ValueError(f"by must be 0 or 1, got {by}")
        self.by = by

        if k_or_tau is not None:
            if isinstance(k_or_tau, int) and not k_or_tau >= 1:
                raise ValueError(f"k must be positive >= 1, got {k_or_tau}")

            if isinstance(k_or_tau, float) and not 0 < k_or_tau < 1:
                raise ValueError(f"tau must be in (0, 1), got {k_or_tau}")
        self.k_or_tau = k_or_tau
        self.reverse = reverse

        if surrogate not in SURROGATES:
            raise ValueError(f"surrogate must be one of {list(SURROGATES.keys())}")
        self.surrogate = surrogate
        self._t = None
        self._t_ind = None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        t, ind = self.threshold(save=False)
        if t is not None:
            t = t.detach().cpu().item()
            ind = ind.detach().cpu().item()

        state.update({"threshold": t, "threshold_index": ind})
        return state

    def load_state_dict(self, state, *args, **kwargs):
        t = state.pop("threshold", None)
        ind = state.pop("threshold_index", None)

        if t is not None:
            self._t = torch.tensor(t)
            self._t_ind = torch.tensor(ind)

        super().load_state_dict(state, *args, **kwargs)
        return

    def threshold(
        self,
        s: Tensor | None = None,
        y: Tensor | None = None,
        save: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if y is None or s is None:
            return self._t, self._t_ind

        if len(y) == 0:
            return self._t, self._t_ind

        t, ind = find_threshold(
            y=torch.flatten(y),
            s=torch.flatten(s),
            by=self.by,
            k_or_tau=self.k_or_tau,
            reverse=self.reverse,
        )

        if save:
            self._t = t.detach().cpu()
            self._t_ind = ind.detach().cpu()

        return t, ind

    def forward(self, s: Tensor, y: Tensor) -> Tensor:
        t, _ = self.threshold(s, y, save=self.training)

        # false-negative rate
        alpha = self.alpha
        surrogate = SURROGATES[self.surrogate]

        fnr = torch.sum((y == 1) * surrogate(t - s)) / torch.sum(y == 1)
        fpr = torch.sum((y == 0) * surrogate(s - t)) / torch.sum(y == 0)

        return alpha * fnr + (1 - alpha) * fpr

    def predict(self, s: Tensor) -> Tensor:
        t, _ = self.threshold()
        return s >= t


class DeepTopPush(ClassificationAtTopObjective):
    def __init__(self, surrogate: str = "hinge", k: int = None):
        super().__init__(
            alpha=1,
            by=0,
            k_or_tau=k,
            reverse=True,
            surrogate=surrogate,
        )

    def __repr__(self):
        surrogate = self.surrogate
        return f"DeepTopPush({surrogate=})"


class PatMat(ClassificationAtTopObjective):
    def __init__(self, tau: float, surrogate: str = "hinge"):
        super().__init__(
            alpha=1,
            by=None,
            k_or_tau=tau,
            reverse=True,
            surrogate=surrogate,
        )

    def __repr__(self):
        tau = self.k_or_tau
        surrogate = self.surrogate
        return f"PatMat({tau=}, {surrogate=})"


class PatMatNP(ClassificationAtTopObjective):
    def __init__(self, tau: float, surrogate: str = "hinge"):
        super().__init__(
            alpha=1,
            by=0,
            k_or_tau=tau,
            reverse=True,
            surrogate=surrogate,
        )

    def __repr__(self):
        tau = self.k_or_tau
        surrogate = self.surrogate
        return f"PatMatNP({tau=}, {surrogate=})"
