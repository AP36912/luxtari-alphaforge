"""
Walk-forward validation: expanding or rolling train window, fixed test horizon.

Enforces time order, optional **embargo gap** between train and test, and
**purges** training rows whose label events overlap any test row's event window.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import numpy as np

from validation.event_bounds import event_bounds_from_horizon
from validation.purge import purge_overlapping_train

WindowMode = Literal["expanding", "rolling"]


class WalkForwardSplitter:
    """
    Walk-forward splits with purge on overlapping label intervals.

    Args:
        test_size: Number of consecutive test rows per split.
        train_size: For ``rolling`` mode, fixed training length. Ignored when ``expanding``.
        window: ``expanding`` grows training through each completed test block; ``rolling`` slides a fixed train window.
        step: For ``rolling`` only: advance of the train window start between splits.
        embargo: Rows skipped between end of training region and start of test (pre-test gap).
        label_horizon: Label lookforward horizon for purge logic.
        initial_train: Training length ``[0, initial_train)`` before the first test window.
        event_start_offset: First future bar in label path (default ``1``).

    Expanding mode advances the test block by ``test_size + embargo`` each split (non-overlapping tests);
    training becomes ``[0, previous_test_end)``. The ``step`` argument is ignored when ``window="expanding"``.
    """

    def __init__(
        self,
        test_size: int,
        *,
        train_size: int | None = None,
        window: WindowMode = "expanding",
        step: int | None = None,
        embargo: int = 0,
        label_horizon: int,
        initial_train: int,
        event_start_offset: int = 1,
    ) -> None:
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if initial_train < 1:
            raise ValueError("initial_train must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        if window == "rolling" and (train_size is None or train_size < 1):
            raise ValueError("rolling window requires train_size >= 1")
        self.test_size = test_size
        self.train_size = train_size
        self.window = window
        self.step = step if step is not None else test_size
        if window == "rolling" and self.step < 1:
            raise ValueError("step must be >= 1")
        self.embargo = embargo
        self.label_horizon = label_horizon
        self.initial_train = initial_train
        self.event_start_offset = event_start_offset

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

        if self.window == "expanding":
            train_end = self.initial_train
            while True:
                test_start = train_end + self.embargo
                test_end = test_start + self.test_size
                if test_end > n:
                    break
                train_idx = np.arange(0, train_end, dtype=np.int64)
                test_idx = np.arange(test_start, test_end, dtype=np.int64)
                train_idx = purge_overlapping_train(train_idx, test_idx, es, ee)
                yield np.sort(train_idx), test_idx
                train_end = test_end

        else:
            # rolling
            assert self.train_size is not None
            t_start = 0
            while True:
                train_end = t_start + self.train_size
                test_start = train_end + self.embargo
                test_end = test_start + self.test_size
                if test_end > n:
                    break
                train_idx = np.arange(t_start, train_end, dtype=np.int64)
                test_idx = np.arange(test_start, test_end, dtype=np.int64)
                train_idx = purge_overlapping_train(train_idx, test_idx, es, ee)
                yield np.sort(train_idx), test_idx
                t_start += self.step

    def get_n_splits(self, X: Any, y: Any = None, groups: Any = None) -> int:
        n = len(X)
        if self.window == "expanding":
            train_end = self.initial_train
            count = 0
            while True:
                test_end = train_end + self.embargo + self.test_size
                if test_end > n:
                    break
                count += 1
                train_end = test_end
            return count
        assert self.train_size is not None
        t_start = 0
        count = 0
        while True:
            train_end = t_start + self.train_size
            test_end = train_end + self.embargo + self.test_size
            if test_end > n:
                break
            count += 1
            t_start += self.step
        return count
