from typing import Iterable

import classification_at_top as cat
import classification_at_top.metrics as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DATA = "data"


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, input, target):
        return F.binary_cross_entropy(torch.sigmoid(input), target.float())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return torch.flatten(x)


def prepare_data(
    objective=None,
    train_batch_size: int = 64,
    test_batch_size: int = 1000,
    positive_classes: int | Iterable[int] = 0,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    pos = torch.tensor(positive_classes)

    # train data
    train = datasets.MNIST(DATA, train=True, download=True, transform=transform)
    train.targets = torch.isin(train.targets, pos)
    train_loader = DataLoader(
        train,
        shuffle=False,
        batch_sampler=cat.StratifiedRandomSampler(
            targets=train.targets,
            pos_label=1,
            batch_size_neg=train_batch_size // 2,
            batch_size_pos=train_batch_size // 2,
            objective=objective,
        ),
    )

    # test data
    test = datasets.MNIST(DATA, train=False, transform=transform)
    test.targets = torch.isin(test.targets, pos)
    test_loader = DataLoader(test, shuffle=True, batch_size=test_batch_size)

    return train_loader, test_loader


def train(model, objective, device, train_loader, optimizer, epoch):
    model.train()
    objective.train()
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = objective(output, target)
        loss.backward()
        optimizer.step()


def test(model, objective, device, test_loader):
    model.eval()
    objective.eval()
    test_loss = 0

    y = []
    s = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += objective(output, target).item()

            y.append(target)
            s.append(output)
    targets = torch.flatten(torch.cat(y, 0))
    scores = torch.flatten(torch.cat(s, 0))

    if getattr(objective, "threshold", None) is not None:
        t, _ = objective.threshold(scores, targets, save=False)
    else:
        t = 0.5

    cm = mt.BinaryConfusionMatrix(threshold=t)
    cm.update(scores, targets)

    print("Test metrics: ")
    print(f"  - loss: {test_loss / len(test_loader.dataset)}")
    print(f"  - threshold: {t}")

    metrics = [
        mt.accuracy,
        mt.balanced_accuracy,
        (mt.true_negative_rate, mt.true_negatives),
        (mt.false_positive_rate, mt.false_positives),
        (mt.true_positive_rate, mt.true_positives),
        (mt.false_negative_rate, mt.false_negatives),
    ]

    for metric in metrics:
        if isinstance(metric, tuple):
            msg = f"  - {metric[0].__name__}: {100 * metric[0](cm): .2f} ({metric[1](cm)})"
        else:
            msg = f"  - {metric.__name__}: {100 * metric(cm): .2f}"
        print(msg)

    m1 = mt.positive_rate_at_top
    m2 = mt.positives_at_top
    print(
        f"  - {m1.__name__}: {100 * m1(scores, targets): .2f} ({m2(scores, targets)})"
    )

    return targets, scores, t, cm
