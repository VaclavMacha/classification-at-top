import random
from math import ceil
from typing import Iterator, List

import torch
from torch import Tensor
from torch.utils.data import Sampler


class StratifiedRandomSampler(Sampler[List[int]]):
    def __init__(
        self,
        targets: Tensor,
        pos_label: int,
        batch_size_neg: int,
        batch_size_pos: int,
        objective: torch.nn.Module | None = None,
        max_iters: int | None = None,
    ) -> None:
        self.batch_size_neg = batch_size_neg
        self.batch_size_pos = batch_size_pos
        self.inds_neg = torch.where(targets != pos_label)[0].tolist()
        self.inds_pos = torch.where(targets == pos_label)[0].tolist()
        self.n_samples = len(targets)
        self.objective = objective
        self.max_iters = max_iters

    @staticmethod
    def _sample(inds: List[int], k: int) -> List[int]:
        if len(inds) < k:
            return random.choices(inds, k=k)
        else:
            return random.sample(inds, k=k)

    def _generate_batch(self) -> Iterator[List[int]]:
        batch = [
            *self._sample(self.inds_neg, k=self.batch_size_neg),
            *self._sample(self.inds_pos, k=self.batch_size_pos),
        ]

        obj = self.objective
        if obj is not None and getattr(obj, "threshold", None) is not None:
            _, t_ind = obj.threshold()
            if t_ind is not None and t_ind not in batch:
                ind = random.randint(0, len(batch) - 1)
                batch[ind] = t_ind.item()

        return batch

    def __len__(self):
        if self.max_iters is not None:
            return self.max_iters

        return ceil(self.n_samples / (self.batch_size_neg + self.batch_size_pos))

    def __iter__(self):
        self.batch_count = 0
        while True:
            if self.batch_count <= len(self):
                self.batch_count += 1
                yield self._generate_batch()
            else:
                break
