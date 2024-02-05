import pytest
from classification_at_top.utilities import (
    find_extrema,
    find_kth,
    find_quantile,
)
from torch import tensor


@pytest.mark.parametrize(
    "x, reverse, expected",
    [
        (range(1, 11), False, 0),
        (range(1, 11), True, 9),
    ],
)
class TestExtrema:
    def test_expected(self, x, reverse, expected):
        actual = find_extrema(tensor(x), reverse)
        assert actual == tensor(expected)

    def test_kth(self, x, reverse, expected):
        t1 = find_extrema(tensor(x), reverse)
        t2 = find_kth(tensor(x), 0, reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "x, k, reverse, expected",
    [
        (range(1, 11), 0, False, 0),
        (range(1, 11), 1, False, 1),
        (range(1, 11), 2, False, 2),
        (range(1, 11), 3, False, 3),
        (range(1, 11), 4, False, 4),
        (range(1, 11), 5, False, 5),
        (range(1, 11), 6, False, 6),
        (range(1, 11), 7, False, 7),
        (range(1, 11), 8, False, 8),
        (range(1, 11), 9, False, 9),
        (range(1, 11), 0, True, 9),
        (range(1, 11), 1, True, 8),
        (range(1, 11), 2, True, 7),
        (range(1, 11), 3, True, 6),
        (range(1, 11), 4, True, 5),
        (range(1, 11), 5, True, 4),
        (range(1, 11), 6, True, 3),
        (range(1, 11), 7, True, 2),
        (range(1, 11), 8, True, 1),
        (range(1, 11), 9, True, 0),
    ],
)
class TestFindKth:
    def test_expected(self, x, k, reverse, expected):
        actual = find_kth(tensor(x), k, reverse)
        assert actual == tensor(expected)

    def test_negation(self, x, k, reverse, expected):
        t1 = find_kth(tensor(x), k, reverse)
        t2 = find_kth(tensor(x), len(x) - k - 1, not reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "x, tau, top, expected",
    [
        (range(1, 12), 0.1, False, 1),
        (range(1, 12), 0.2, False, 2),
        (range(1, 12), 0.3, False, 3),
        (range(1, 12), 0.4, False, 4),
        (range(1, 12), 0.5, False, 5),
        (range(1, 12), 0.6, False, 6),
        (range(1, 12), 0.7, False, 7),
        (range(1, 12), 0.8, False, 8),
        (range(1, 12), 0.9, False, 9),
        (range(1, 12), 0.1, True, 9),
        (range(1, 12), 0.2, True, 8),
        (range(1, 12), 0.3, True, 7),
        (range(1, 12), 0.4, True, 6),
        (range(1, 12), 0.5, True, 5),
        (range(1, 12), 0.6, True, 4),
        (range(1, 12), 0.7, True, 3),
        (range(1, 12), 0.8, True, 2),
        (range(1, 12), 0.9, True, 1),
    ],
)
class TestFindQuantile:
    def test_expected(self, x, tau, top, expected):
        actual = find_quantile(tensor(x, dtype=float), tau, top)
        assert actual == tensor(expected)

    def test_negation(self, x, tau, top, expected):
        t1 = find_quantile(tensor(x, dtype=float), tau, top)
        t2 = find_quantile(tensor(x, dtype=float), 1 - tau, not top)
        assert t1 == t2
