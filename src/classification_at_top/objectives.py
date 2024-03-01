from typing import Union

import torch
from attrs import converters, define, field, validators
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


@define(frozen=False)
class ClassificationAtTopObjective(Module):
    alpha: float = field(
        converter=float,
        validator=[
            validators.instance_of(Number),
            validators.ge(0),
            validators.le(1),
        ],
    )
    by: int | None = field(
        converter=converters.optional(int),
        validator=validators.optional(
            [
                validators.instance_of(int),
                validators.in_([0, 1]),
            ]
        ),
    )
    k_or_tau: Number | None = field(
        validator=validators.optional(validators.instance_of(Number)),
    )
    reverse: bool = field(
        validator=validators.instance_of(bool),
    )
    surrogate: str = field(
        default="hinge",
        validator=[
            validators.instance_of(str),
            validators.in_(["hinge", "quadratic_hinge"]),
        ],
    )

    def __attrs_pre_init__(self):
        super().__init__()
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
        y: Tensor | None = None,
        s: Tensor | None = None,
        save: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if y is None or s is None:
            return self._t, self._t_ind

        t, ind = find_threshold(y, s, self.by, self.k_or_tau, self.reverse)

        if save:
            self._t = t.detach().cpu()
            self._t_ind = ind.detach().cpu()

        return t, ind

    def forward(self, s: Tensor, y: Tensor) -> Tensor:
        t, _ = self.threshold(y, s, save=self.training)

        # false-negative rate
        alpha = self.alpha
        surrogate = SURROGATES[self.surrogate]

        fnr = torch.sum((y == 1) * surrogate(t - s)) / torch.sum(y == 1)
        fpr = torch.sum((y == 0) * surrogate(s - t)) / torch.sum(y == 0)

        return alpha * fnr + (1 - alpha) * fpr

    def predict(self, s: Tensor) -> Tensor:
        t, _ = self.threshold()
        return s >= t


@define(frozen=False)
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


@define(frozen=False)
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


@define(frozen=False)
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
