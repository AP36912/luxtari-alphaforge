"""
Purged K-Fold cross-validation (sequential folds, event overlap purge, embargo).

Data must be **time-ordered** on rows. Each fold uses one contiguous test block;
training is all other rows minus overlapping label windows and optional embargo
after the test block.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from validation.event_bounds import event_bounds_from_horizon
from validation.purge import apply_embargo, purge_overlapping_train


class PurgedKFold:
    """
    Purged K-fold splitter.

    Args:
        n_splits: Number of contiguous test folds.
        label_horizon: Bars after each decision used by the label (e.g. triple-barrier ``max_bars``).
        embargo: Training row indices in ``(test_max, test_max + embargo]`` are dropped.
        event_start_offset: First future bar used by the label (default ``1``, matching ``labeling``).
    """

    def __init__(
        self,
        n_splits: int,
        *,
        label_horizon: int,
        embargo: int = 0,
        event_start_offset: int = 1,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.event_start_offset = event_start_offset

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n < self.n_splits:
            raise ValueError("Not enough samples for n_splits")
        es, ee = event_bounds_from_horizon(
            n,
            self.label_horizon,
            event_start_offset=self.event_start_offset,
        )

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

        for k in range(self.n_splits):
            t0, t1 = boundaries[k], boundaries[k + 1]
            test_idx = np.arange(t0, t1, dtype=np.int64)
            train_idx = np.setdiff1d(np.arange(n, dtype=np.int64), test_idx)
            train_idx = purge_overlapping_train(train_idx, test_idx, es, ee)
            train_idx = apply_embargo(train_idx, test_idx, self.embargo)
            train_idx = np.sort(train_idx)
            yield train_idx, test_idx
