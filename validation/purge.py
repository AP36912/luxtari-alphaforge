"""
Purge training indices whose label event intervals overlap test samples.

Optionally apply post-test **embargo**: drop training decisions that fall too
soon after the last test decision (microstructure / serial correlation guard).
"""

from __future__ import annotations

import numpy as np


def purge_overlapping_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    event_start: np.ndarray,
    event_end: np.ndarray,
) -> np.ndarray:
    """
    Remove training samples whose event interval intersects any test sample's interval.

    Args:
        train_idx: Candidate training row indices.
        test_idx: Test row indices.
        event_start, event_end: Length ``n`` arrays (full dataset), inclusive bounds.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if len(test_idx) == 0:
        return train_idx
    ts = event_start[test_idx]
    te = event_end[test_idx]
    keep_mask = np.ones(len(train_idx), dtype=bool)
    for k, i in enumerate(train_idx):
        s = event_start[i]
        e = event_end[i]
        over = (s <= te) & (ts <= e)
        if np.any(over):
            keep_mask[k] = False
    return train_idx[keep_mask]


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """
    Drop training indices in ``(test_max, test_max + embargo]`` (exclusive of test).

    Only applies to training observations **after** the test block; typical when
    train includes future folds relative to a middle test chunk.
    """
    if embargo <= 0 or len(test_idx) == 0:
        return train_idx
    train_idx = np.asarray(train_idx, dtype=np.int64)
    te_max = int(np.max(test_idx))
    # Remove train decisions that occur too soon after the end of the test window
    bad = (train_idx > te_max) & (train_idx <= te_max + embargo)
    return train_idx[~bad]
