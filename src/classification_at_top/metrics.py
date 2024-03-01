from torch import Tensor
from torcheval.metrics import BinaryConfusionMatrix


def negatives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[0, :].sum().item()


def positives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[1, :].sum().item()


def true_negatives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[0, 0].item()


def false_positives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[0, 1].item()


def true_positives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[1, 1].item()


def false_negatives(cm: BinaryConfusionMatrix) -> int:
    return cm.compute()[1, 0].item()


def true_negative_rate(cm: BinaryConfusionMatrix) -> float:
    return true_negatives(cm) / negatives(cm)


def false_positive_rate(cm: BinaryConfusionMatrix) -> float:
    return 1 - true_negative_rate(cm)


def true_positive_rate(cm: BinaryConfusionMatrix) -> float:
    return true_positives(cm) / positives(cm)


def false_negative_rate(cm: BinaryConfusionMatrix) -> float:
    return 1 - true_positive_rate(cm)


def accuracy(cm: BinaryConfusionMatrix) -> float:
    numerator = true_positives(cm) + true_negatives(cm)
    denominator = cm.compute().sum().item()

    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def balanced_accuracy(cm: BinaryConfusionMatrix) -> float:
    return (true_negative_rate(cm) + true_positive_rate(cm)) / 2


def positives_at_top(targets: Tensor, scores: Tensor) -> int:
    return (scores[targets == 1] >= scores[targets == 0].max()).sum().item()


def positive_rate_at_top(targets: Tensor, scores: Tensor) -> int:
    numerator = positives_at_top(targets, scores)
    denominator = (targets == 1).sum().item()

    if denominator == 0:
        return 0
    else:
        return numerator / denominator
