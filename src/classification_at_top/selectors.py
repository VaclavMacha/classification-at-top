from abc import ABC, abstractmethod

import attrs.validators as validators
import numpy as np
from attrs import define, field


class AbstractSelector(ABC):
    @abstractmethod
    def indices(self, y: np.ndarray) -> np.ndarray:
        """
        Abstract method that returns the indices of selected elements in the input array.
        This method must be implemented by subclasses.

        Args:
            y (np.ndarray): The input array.

        Returns:
            np.ndarray: The indices of selected elements.
        """
        pass

    def select(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Selects elements from the input array based on the indices returned by the `indices` method.

        Args:
            y (np.ndarray): The input array.
            s (np.ndarray): The array to select elements from.

        Returns:
            np.ndarray: The selected elements from the input array.
        """
        inds = self.indices(y)
        return s[inds], inds


@define(frozen=True)
class All(AbstractSelector):
    """
    A selector that returns all indices of the input array.
    """

    def indices(self, y: np.ndarray) -> np.ndarray:
        """
        Returns all indices of the input array.

        Args:
            y (np.ndarray): The input array.

        Returns:
            np.ndarray: All indices of the input array.
        """
        return np.arange(y.size).reshape(y.shape)


@define(frozen=True)
class Positives(AbstractSelector):
    """
    A selector that returns the indices of positive elements in the input array.

    Attributes:
        positive_label (int): The label representing positive elements.
    """

    positive_label: int = field(
        default=1,
        validator=validators.instance_of(int),
    )

    def indices(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the indices of positive elements in the input array.

        Args:
            y (np.ndarray): The input array.

        Returns:
            np.ndarray: The indices of positive elements.
        """
        return np.where(y == self.positive_label)[0]


@define(frozen=True)
class Negatives(AbstractSelector):
    """
    A selector that returns the indices of negative elements in the input array.

    Attributes:
        negative_label (int): The label representing negative elements.
    """

    negative_label: int = field(
        default=0,
        validator=validators.instance_of(int),
    )

    def indices(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the indices of negative elements in the input array.

        Args:
            y (np.ndarray): The input array.

        Returns:
            np.ndarray: The indices of negative elements.
        """
        return np.where(y == self.negative_label)[0]
