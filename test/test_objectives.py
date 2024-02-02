import pytest
import torch
from classification_at_top.objectives import hinge, quadratic_hinge


@pytest.mark.parametrize(
    "x, scale, expected",
    [
        ([-1 / s - 1, 0, -1 / s + 1], s, [0, 1, s])
        for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ],
)
def test_hinge(x, scale, expected):
    assert torch.allclose(
        hinge(torch.tensor(x), scale),
        torch.tensor(expected),
    )


@pytest.mark.parametrize(
    "x, scale, expected",
    [
        ([-1 / s - 1, 0, -1 / s + 1], s, [0, 1, s**2])
        for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ],
)
def test_quadratic_hinge(x, scale, expected):
    assert torch.allclose(
        quadratic_hinge(torch.tensor(x), scale),
        torch.tensor(expected),
    )
