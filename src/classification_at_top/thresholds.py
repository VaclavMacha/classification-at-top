from abc import ABC, abstractmethod
from typing import Tuple

import attrs.converters as converters
import attrs.validators as validators
import numpy as np
from attrs import define, field

from .selectors import AbstractSelector, All
from .utilities import (
    ArrayLike,
    Number,
    find_kth,
    find_maximum,
    find_minimum,
    find_quantile,
    to_numpy,
)


class AbstractThresholdType(ABC):
    """
    Abstract class representing a classification threshold.
    """

    @abstractmethod
    def find(self, y: ArrayLike, s: ArrayLike) -> Tuple[Number, int]:
        """
        Finds the threshold and its index in the given array of scores `s`. This method must be implemented by subclasses.

        Args:
            y (ArrayLike): The target values.
            s (ArrayLike): The score values.

        Returns:
            Tuple[Number, int]: A tuple containing the threshold and its index.
        """
        pass


@define(frozen=True)
class Maximum(AbstractThresholdType):
    """
    A class representing a classification threshold equal to the maximum sample.

    Attributes:
        selector (AbstractSelector): The selector that decides whether the threshold is
            computed from all, positive, or negative samples.
    """

    selector: AbstractSelector = field(
        default=All(),
        validator=validators.instance_of(AbstractSelector),
    )

    def find(self, y: ArrayLike, s: ArrayLike) -> Tuple[Number, int]:
        """
        Finds the maximum and its index from in given array of scores `s`.
        Only scores selected by the selector are considered.

        Args:
            y (ArrayLike): The target values.
            s (ArrayLike): The score values.

        Returns:
            Tuple[Number, int]: A tuple containing the threshold and its index.
        """
        s_selected, inds = self.selector.select(to_numpy(y), to_numpy(s))
        t, t_ind = find_maximum(s_selected)

        return t, inds[t_ind]


@define(frozen=True)
class Minimum(AbstractThresholdType):
    """
    A class representing a classification threshold equal to the minimum sample.

    Attributes:
        selector (AbstractSelector): The selector that decides whether the threshold is
            computed from all, positive, or negative samples.
    """

    selector: AbstractSelector = field(
        default=All(),
        validator=validators.instance_of(AbstractSelector),
    )

    def find(self, y: ArrayLike, s: ArrayLike) -> Tuple[Number, int]:
        """
        Finds the minimum and its index in the given array of scores `s`.
        Only scores selected by the selector are considered.

        Args:
            y (ArrayLike): The target values.
            s (ArrayLike): The score values.

        Returns:
            Tuple[Number, int]: A tuple containing the threshold and its index.
        """
        s_selected, inds = self.selector.select(to_numpy(y), to_numpy(s))
        t, t_ind = find_minimum(s_selected)

        return t, inds[t_ind]


@define(frozen=True)
class Kth(AbstractThresholdType):
    """
    A class representing a classification threshold equal to the k-th largest/smallest
    sample.

    Attributes:
        k (int): The index of the element to find in the sorted array.
        reverse (bool): If True, the array is sorted in descending order. If False, the
            array is sorted in ascending order. Defaults to True.
        selector (AbstractSelector): The selector that decides whether the threshold is
            computed from all, positive, or negative samples.
    """

    k: int = field(
        converter=int,
        validator=validators.ge(0),
    )
    reverse: bool = field(
        default=True,
        converter=converters.to_bool,
    )
    selector: AbstractSelector = field(
        default=All(),
        validator=validators.instance_of(AbstractSelector),
    )

    def find(self, y: ArrayLike, s: ArrayLike) -> Tuple[Number, int]:
        """
        Finds the k-th largest/smallest sample and its index from in given array of scores
        `s`. Only scores selected by the selector are considered.

        Args:
            y (ArrayLike): The target values.
            s (ArrayLike): The score values.

        Returns:
            Tuple[Number, int]: A tuple containing the threshold and its index.
        """
        s_selected, inds = self.selector.select(to_numpy(y), to_numpy(s))
        t, t_ind = find_kth(s_selected, self.k, self.reverse)
        return t, inds[t_ind]


@define(frozen=True)
class Quantile(AbstractThresholdType):
    """
    A class representing a classification threshold equal to the top/bottom quantile.

    Attributes:
        tau (float): The quantile value to use for thresholding. Must be between 0 and 1.
        top (bool): If True, finds the quantile from the top of the array. If False, finds
            the quantile from the bottom. Defaults to False.
        selector (AbstractSelector): The selector used to select the data for thresholding.
    """

    tau: float = field(
        converter=float,
        validator=[validators.ge(0), validators.le(1)],
    )
    top: bool = field(
        default=True,
        converter=converters.to_bool,
    )
    selector: AbstractSelector = field(
        default=All(),
        validator=validators.instance_of(AbstractSelector),
    )

    def find(self, y: ArrayLike, s: ArrayLike) -> Tuple[Number, int]:
        """
        Finds the top/bottom tau-quantile and its index from in given array of scores
        `s`. Only scores selected by the selector are considered.

        Args:
            y (ArrayLike): The target values.
            s (ArrayLike): The score values.

        Returns:
            Tuple[Number, int]: A tuple containing the threshold and its index
        """
        s_selected, inds = self.selector.select(to_numpy(y), to_numpy(s))
        t, t_ind = find_quantile(s_selected, self.tau, self.top)
        return t, inds[t_ind]
