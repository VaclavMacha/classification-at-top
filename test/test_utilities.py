import numpy as np
import pytest
import torch
from classification_at_top.utilities import find_kth, find_quantile, to_numpy, to_torch


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        ([1, 2, 3], np.array([1, 2, 3])),
        ((1, 2, 3), np.array([1, 2, 3])),
        (torch.tensor([1, 2, 3]), np.array([1, 2, 3])),
    ],
)
def test_to_numpy(x, expected):
    actual = to_numpy(x)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize(
    "device, x, expected",
    [
        ("cpu", [1, 2, 3], torch.tensor([1, 2, 3], device="cpu")),
        ("cpu", (1, 2, 3), torch.tensor([1, 2, 3], device="cpu")),
        ("cpu", torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3], device="cpu")),
        (None, [1, 2, 3], torch.tensor([1, 2, 3], device="cpu")),
        (None, (1, 2, 3), torch.tensor([1, 2, 3], device="cpu")),
        (None, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3], device="cpu")),
    ],
)
def test_to_torch(device, x, expected):
    actual = to_torch(x, device=device)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "x, k, reverse, expected",
    [
        (range(1, 11), 0, False, (1, 0)),
        (range(1, 11), 1, False, (2, 1)),
        (range(1, 11), 2, False, (3, 2)),
        (range(1, 11), 3, False, (4, 3)),
        (range(1, 11), 4, False, (5, 4)),
        (range(1, 11), 5, False, (6, 5)),
        (range(1, 11), 6, False, (7, 6)),
        (range(1, 11), 7, False, (8, 7)),
        (range(1, 11), 8, False, (9, 8)),
        (range(1, 11), 9, False, (10, 9)),
        (range(1, 11), 0, True, (10, 9)),
        (range(1, 11), 1, True, (9, 8)),
        (range(1, 11), 2, True, (8, 7)),
        (range(1, 11), 3, True, (7, 6)),
        (range(1, 11), 4, True, (6, 5)),
        (range(1, 11), 5, True, (5, 4)),
        (range(1, 11), 6, True, (4, 3)),
        (range(1, 11), 7, True, (3, 2)),
        (range(1, 11), 8, True, (2, 1)),
        (range(1, 11), 9, True, (1, 0)),
    ],
)
class TestFindKth:
    def test_expected(self, x, k, reverse, expected):
        actual = find_kth(to_numpy(x), k, reverse)
        assert actual == expected

    def test_negation(self, x, k, reverse, expected):
        t1 = find_kth(to_numpy(x), k, reverse)
        t2 = find_kth(to_numpy(x), len(x) - k - 1, not reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "x, tau, top, expected",
    [
        (range(1, 12), 0.0, False, (1, 0)),
        (range(1, 12), 0.1, False, (2, 1)),
        (range(1, 12), 0.2, False, (3, 2)),
        (range(1, 12), 0.3, False, (4, 3)),
        (range(1, 12), 0.4, False, (5, 4)),
        (range(1, 12), 0.5, False, (6, 5)),
        (range(1, 12), 0.6, False, (7, 6)),
        (range(1, 12), 0.7, False, (8, 7)),
        (range(1, 12), 0.8, False, (9, 8)),
        (range(1, 12), 0.9, False, (10, 9)),
        (range(1, 12), 1.0, False, (11, 10)),
        (range(1, 12), 0.0, True, (11, 10)),
        (range(1, 12), 0.1, True, (10, 9)),
        (range(1, 12), 0.2, True, (9, 8)),
        (range(1, 12), 0.3, True, (8, 7)),
        (range(1, 12), 0.4, True, (7, 6)),
        (range(1, 12), 0.5, True, (6, 5)),
        (range(1, 12), 0.6, True, (5, 4)),
        (range(1, 12), 0.7, True, (4, 3)),
        (range(1, 12), 0.8, True, (3, 2)),
        (range(1, 12), 0.9, True, (2, 1)),
        (range(1, 12), 1.0, True, (1, 0)),
    ],
)
class TestFindQuantile:
    def test_expected(self, x, tau, top, expected):
        actual = find_quantile(to_numpy(x), tau, top)
        assert actual == expected

    def test_negation(self, x, tau, top, expected):
        t1 = find_quantile(to_numpy(x), tau, top)
        t2 = find_quantile(to_numpy(x), 1 - tau, not top)
        assert t1 == t2
