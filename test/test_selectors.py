import numpy as np
import pytest
from classification_at_top.selectors import All, Negatives, Positives
from classification_at_top.utilities import to_numpy


@pytest.mark.parametrize(
    "selector, y, s, inds",
    [
        (All(), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4]),
        (Negatives(), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [2, 4]),
        (Negatives(0), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [2, 4]),
        (Negatives(1), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [0, 1, 3]),
        (Negatives(1234), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], []),
        (Positives(), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [0, 1, 3]),
        (Positives(1), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [0, 1, 3]),
        (Positives(0), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], [2, 4]),
        (Positives(1234), [1, 1, 0, 1, 0], [1, 2, 3, 4, 5], []),
    ],
)
class TestSelectors:
    def test_indices(self, selector, y, s, inds):
        expected = to_numpy(inds)
        actual = selector.indices(to_numpy(y))
        assert np.array_equal(expected, actual)

    def test_select(self, selector, y, s, inds):
        expected = to_numpy(s)[inds]
        actual = selector.select(to_numpy(y), to_numpy(s))[0]
        assert np.array_equal(expected, actual)
