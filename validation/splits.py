"""
Compatibility helpers for time-series splits (purged K-fold indices, etc.).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from validation.purged_kfold import PurgedKFold


def purged_kfold_indices(
    n_samples: int,
    n_splits: int,
    *,
    label_horizon: int,
    embargo: int = 0,
    event_start_offset: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield ``(train_idx, test_idx)`` for purged K-fold on ``range(n_samples)``.

    Prefer :class:`validation.purged_kfold.PurgedKFold` when working with arrays directly.
    """
    pkf = PurgedKFold(
        n_splits,
        label_horizon=label_horizon,
        embargo=embargo,
        event_start_offset=event_start_offset,
    )
    X = np.empty(n_samples)
    yield from pkf.split(X)
