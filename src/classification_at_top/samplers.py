import random
from inspect import cleandoc
from math import ceil
from typing import Iterable, Iterator, List, Union

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, Sampler

from .objectives import ClassificationAtTopObjective


class ClassificationAtTopBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        objective: ClassificationAtTopObjective,
    ) -> None:
        super().__init__(sampler, batch_size, False)

        if not isinstance(objective, ClassificationAtTopObjective):
            raise ValueError(
                cleandoc(
                    f"""Objective should be an instance of ClassificationAtTopObjective, but got objective={type(objective)=}
                    """
                )
            )
        self.objective = objective

    def _add_threshold_ind(self, batch: List[int]) -> List[int]:
        _, t_ind = self.objective.threshold()
        if t_ind is not None and t_ind not in batch:
            ind = random.randint(0, len(batch) - 1)
            batch[ind] = t_ind.item()
        return batch

    def __iter__(self) -> Iterator[List[int]]:
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield self._add_threshold_ind(batch)
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield self._add_threshold_ind(batch[:idx_in_batch])


class StratifiedRandomSampler(Sampler[List[int]]):
    def __init__(
        self,
        targets: Tensor,
        pos_label: int,
        batch_size_neg: int,
        batch_size_pos: int,
        objective: ClassificationAtTopObjective | None = None,
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

        if self.objective is None:
            _, t_ind = self.objective.threshold()
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
