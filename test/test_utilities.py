import numpy as np
import pytest
import torch
from classification_at_top.utilities import to_numpy, to_torch


@pytest.mark.parametrize(
    "x,expected",
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
    "device,x,expected",
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
        *[(np.array(range(1, 11)), k, False, (k + 1, k)) for k in range(10)],
        *[(np.array(range(1, 11)), k, True, (10 - k, 9 - k)) for k in range(10)],
    ],
)
def test_find_kth(x, k, reverse, expected):
    actual = find_kth(x, k, reverse)
    assert actual == expected

