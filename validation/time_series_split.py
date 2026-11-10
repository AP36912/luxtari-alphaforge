"""
Time-series cross-validation with gap (embargo), plus event-interval purging.

Wraps :class:`sklearn.model_selection.TimeSeriesSplit` (chronological train/test,
optional ``max_train_size`` / ``test_size``) and removes training samples whose
label event windows overlap any test sample (same contract as :class:`PurgedKFold`).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from validation.event_bounds import event_bounds_from_horizon
from validation.purge import purge_overlapping_train


class PurgedTimeSeriesSplit:
    """
    sklearn ``TimeSeriesSplit`` + purge of overlapping label intervals.

    Args:
        n_splits: Number of splits (see sklearn).
        embargo: Gap in **rows** between end of train and start of test (sklearn ``gap``).
        label_horizon: Horizon used to build event intervals for purging.
        event_start_offset: First future bar in the label path (default ``1``).
        max_train_size: Optional cap on training size (sklearn).
        test_size: Optional fixed test set size (sklearn); if None, sklearn infers sizes.
    """

    def __init__(
        self,
        n_splits: int,
        *,
        embargo: int = 0,
        label_horizon: int,
        event_start_offset: int = 1,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        self.n_splits = n_splits
        self.embargo = embargo
        self.label_horizon = label_horizon
        self.event_start_offset = event_start_offset
        self._tsc = TimeSeriesSplit(
            n_splits=n_splits,
            gap=embargo,
            max_train_size=max_train_size,
            test_size=test_size,
        )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        es, ee = event_bounds_from_horizon(
            n,
            self.label_horizon,
            event_start_offset=self.event_start_offset,
        )
        for train_idx, test_idx in self._tsc.split(X):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            train_idx = purge_overlapping_train(train_idx, test_idx, es, ee)
            yield np.sort(train_idx), test_idx
