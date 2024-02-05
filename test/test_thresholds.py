import pytest
import torch
from classification_at_top.thresholds import FindThreshold, find_threshold
from classification_at_top.utilities import (
    find_extrema,
    find_kth,
    find_quantile,
    to_numpy,
)


@pytest.fixture
def scores_dict() -> dict:
    return {
        "labels": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "scores": list(range(1, 11)),
        "scores_neg": [3, 5, 7, 8, 10],
        "scores_pos": [1, 2, 4, 6, 9],
    }


@pytest.mark.parametrize(
    "by, reverse, key, expected",
    [
        (None, False, "scores", (1, 0)),
        (0, False, "scores_neg", (3, 2)),
        (1, False, "scores_pos", (1, 0)),
        (None, True, "scores", (10, 9)),
        (0, True, "scores_neg", (10, 9)),
        (1, True, "scores_pos", (9, 8)),
    ],
)
class TestExtrema:
    def test_expected(self, by, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        actual = find_threshold(y=y, s=s, by=by, k_or_tau=None, reverse=reverse)
        assert actual == expected

    def test_index(self, by, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        t, t_ind = find_threshold(y=y, s=s, by=by, k_or_tau=None, reverse=reverse)
        assert t == s[t_ind]

    def test_value(self, by, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])
        s_selected = to_numpy(fix[key])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=None, reverse=reverse)[0]
        t2 = find_extrema(x=s_selected, reverse=reverse)[0]
        assert t1 == t2

    def test_kth(self, by, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=None, reverse=reverse)
        t2 = find_threshold(y=y, s=s, by=by, k_or_tau=0, reverse=reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "by, k, reverse, key, expected",
    [
        (None, 1, False, "scores", (2, 1)),
        (0, 1, False, "scores_neg", (5, 4)),
        (1, 1, False, "scores_pos", (2, 1)),
        (None, 1, True, "scores", (9, 8)),
        (0, 1, True, "scores_neg", (8, 7)),
        (1, 1, True, "scores_pos", (6, 5)),
    ],
)
class TestKth:
    def test_expected(self, by, k, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        actual = find_threshold(y=y, s=s, by=by, k_or_tau=k, reverse=reverse)
        assert actual == expected

    def test_index(self, by, k, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        t, t_ind = find_threshold(y=y, s=s, by=by, k_or_tau=k, reverse=reverse)
        assert t == s[t_ind]

    def test_value(self, by, k, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])
        s_selected = to_numpy(fix[key])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=k, reverse=reverse)[0]
        t2 = find_kth(x=s_selected, k=k, reverse=reverse)[0]
        assert t1 == t2

    def test_negation(self, by, k, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])
        n = len(fix[key])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=k, reverse=reverse)
        t2 = find_threshold(y=y, s=s, by=by, k_or_tau=n - k - 1, reverse=not reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "by, tau, reverse, key, expected",
    [
        (None, 0.8, False, "scores", (8, 7)),
        (0, 0.8, False, "scores_neg", (8, 7)),
        (1, 0.8, False, "scores_pos", (6, 5)),
        (None, 0.2, True, "scores", (8, 7)),
        (0, 0.2, True, "scores_neg", (8, 7)),
        (1, 0.2, True, "scores_pos", (6, 5)),
    ],
)
class TestQuantile:
    def test_expected(self, by, tau, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        actual = find_threshold(y=y, s=s, by=by, k_or_tau=tau, reverse=reverse)
        assert actual == expected

    def test_index(self, by, tau, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        t, t_ind = find_threshold(y=y, s=s, by=by, k_or_tau=tau, reverse=reverse)
        assert t == s[t_ind]

    def test_value(self, by, tau, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])
        s_selected = to_numpy(fix[key])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=tau, reverse=reverse)[0]
        t2 = find_quantile(x=s_selected, tau=tau, reverse=reverse)[0]
        assert t1 == t2

    def test_negation(self, by, tau, reverse, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        y = to_numpy(fix["labels"])
        s = to_numpy(fix["scores"])

        t1 = find_threshold(y=y, s=s, by=by, k_or_tau=tau, reverse=reverse)
        t2 = find_threshold(y=y, s=s, by=by, k_or_tau=1 - tau, reverse=not reverse)
        assert t1 == t2


@pytest.mark.parametrize(
    "by, k_or_tau, reverse",
    [
        (None, None, False),
        (0, None, False),
        (1, None, False),
        (None, None, True),
        (0, None, True),
        (1, None, True),
        (None, 1, False),
        (0, 1, False),
        (1, 1, False),
        (None, 1, True),
        (0, 1, True),
        (1, 1, True),
        (None, 0.8, False),
        (0, 0.8, False),
        (1, 0.8, False),
        (None, 0.2, True),
        (0, 0.2, True),
        (1, 0.2, True),
    ],
)
class TestGradient:
    def test_expected(self, by, k_or_tau, reverse, request):
        fix = request.getfixturevalue("scores_dict")
        y = torch.tensor(fix["labels"], requires_grad=False)
        s = torch.tensor(fix["scores"], requires_grad=True, dtype=torch.float64)

        assert torch.autograd.gradcheck(
            FindThreshold.apply, (y, s, by, k_or_tau, reverse)
        )
