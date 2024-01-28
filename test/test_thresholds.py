import pytest
from classification_at_top.selectors import All, Negatives, Positives
from classification_at_top.thresholds import Kth, Maximum, Minimum, Quantile
from classification_at_top.utilities import (
    find_kth,
    find_maximum,
    find_minimum,
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
    "selector, key, expected",
    [
        (All(), "scores", (10, 9)),
        (Negatives(), "scores_neg", (10, 9)),
        (Negatives(0), "scores_neg", (10, 9)),
        (Negatives(1), "scores_pos", (9, 8)),
        (Positives(), "scores_pos", (9, 8)),
        (Positives(1), "scores_pos", (9, 8)),
        (Positives(0), "scores_neg", (10, 9)),
    ],
)
class TestMaximum:
    def test_expected(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Maximum(selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual == expected

    def test_value(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Maximum(selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual[0] == find_maximum(to_numpy(fix[key]))[0]

    def test_kth(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold1 = Maximum(selector=selector)
        threshold2 = Kth(k=0, reverse=True, selector=selector)

        t1 = threshold1.find(fix["labels"], fix["scores"])
        t2 = threshold2.find(fix["labels"], fix["scores"])
        assert t1 == t2


@pytest.mark.parametrize(
    "selector, key, expected",
    [
        (All(), "scores", (1, 0)),
        (Negatives(), "scores_neg", (3, 2)),
        (Negatives(0), "scores_neg", (3, 2)),
        (Negatives(1), "scores_pos", (1, 0)),
        (Positives(), "scores_pos", (1, 0)),
        (Positives(1), "scores_pos", (1, 0)),
        (Positives(0), "scores_neg", (3, 2)),
    ],
)
class TestMinimum:
    def test_expected(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Minimum(selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual == expected

    def test_value(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Minimum(selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual[0] == find_minimum(to_numpy(fix[key]))[0]

    def test_kth(self, selector, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold1 = Minimum(selector=selector)
        threshold2 = Kth(k=0, reverse=False, selector=selector)

        t1 = threshold1.find(fix["labels"], fix["scores"])
        t2 = threshold2.find(fix["labels"], fix["scores"])
        assert t1 == t2


@pytest.mark.parametrize(
    "selector, k, rev, key, expected",
    [
        (All(), 1, False, "scores", (2, 1)),
        (Negatives(), 1, False, "scores_neg", (5, 4)),
        (Negatives(0), 1, False, "scores_neg", (5, 4)),
        (Negatives(1), 1, False, "scores_pos", (2, 1)),
        (Positives(), 1, False, "scores_pos", (2, 1)),
        (Positives(1), 1, False, "scores_pos", (2, 1)),
        (Positives(0), 1, False, "scores_neg", (5, 4)),
        (All(), 1, True, "scores", (9, 8)),
        (Negatives(), 1, True, "scores_neg", (8, 7)),
        (Negatives(0), 1, True, "scores_neg", (8, 7)),
        (Negatives(1), 1, True, "scores_pos", (6, 5)),
        (Positives(), 1, True, "scores_pos", (6, 5)),
        (Positives(1), 1, True, "scores_pos", (6, 5)),
        (Positives(0), 1, True, "scores_neg", (8, 7)),
    ],
)
class TestKth:
    def test_expected(self, selector, k, rev, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Kth(k=k, reverse=rev, selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual == expected

    def test_value(self, selector, k, rev, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Kth(k=k, reverse=rev, selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual[0] == find_kth(to_numpy(fix[key]), k, rev)[0]


@pytest.mark.parametrize(
    "selector, tau, top, key, expected",
    [
        (All(), 0.8, False, "scores", (8, 7)),
        (Negatives(), 0.8, False, "scores_neg", (8, 7)),
        (Negatives(0), 0.8, False, "scores_neg", (8, 7)),
        (Negatives(1), 0.8, False, "scores_pos", (6, 5)),
        (Positives(), 0.8, False, "scores_pos", (6, 5)),
        (Positives(1), 0.8, False, "scores_pos", (6, 5)),
        (Positives(0), 0.8, False, "scores_neg", (8, 7)),
        (All(), 0.2, True, "scores", (9, 8)),
        (Negatives(), 0.2, True, "scores_neg", (10, 9)),
        (Negatives(0), 0.2, True, "scores_neg", (10, 9)),
        (Negatives(1), 0.2, True, "scores_pos", (9, 8)),
        (Positives(), 0.2, True, "scores_pos", (9, 8)),
        (Positives(1), 0.2, True, "scores_pos", (9, 8)),
        (Positives(0), 0.2, True, "scores_neg", (10, 9)),
    ],
)
class TestQuantile:
    def test_expected(self, selector, tau, top, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Quantile(tau=tau, top=top, selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual == expected

    def test_value(self, selector, tau, top, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold = Quantile(tau=tau, top=top, selector=selector)

        actual = threshold.find(fix["labels"], fix["scores"])
        assert actual[0] == find_quantile(to_numpy(fix[key]), tau, top)[0]

    def test_negation(self, selector, tau, top, key, expected, request):
        fix = request.getfixturevalue("scores_dict")
        threshold1 = Quantile(tau=tau, top=top, selector=selector)
        threshold2 = Quantile(tau=1 - tau, top=not top, selector=selector)

        t1 = threshold1.find(fix["labels"], fix["scores"])
        t2 = threshold2.find(fix["labels"], fix["scores"])
        assert t1 == t2
