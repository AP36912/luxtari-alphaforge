"""
Label / event time bounds per sample (integer row indices).

For triple-barrier style labels at decision index ``t``, outcome paths use bars
``t+1 .. t+label_horizon`` inclusive on the **bar index** axis. These intervals
drive purging so training samples do not share outcome windows with test samples.
"""

from __future__ import annotations

import numpy as np


def decision_indices(n_samples: int) -> np.ndarray:
    """Row positions ``0 .. n-1`` as decision-time order (already time-sorted)."""
    return np.arange(n_samples, dtype=np.int64)


def event_bounds_from_horizon(
    n_samples: int,
    label_horizon: int,
    *,
    event_start_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return inclusive ``(event_start, event_end)`` per sample.

    Default matches triple-barrier scanning from ``t+1`` for ``label_horizon``
    future bars: interval ``[t+event_start_offset, t+event_start_offset+label_horizon-1]``.

    Args:
        n_samples: Number of rows (time-ordered).
        label_horizon: Count of future bars in the label definition (e.g. ``max_bars``).
        event_start_offset: First future bar offset (default ``1``).
    """
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    if event_start_offset < 1:
        raise ValueError("event_start_offset must be >= 1")
    t = decision_indices(n_samples)
    start = t + event_start_offset
    end = t + event_start_offset + label_horizon - 1
    return start.astype(np.int64), end.astype(np.int64)


def intervals_overlap(
    s1: np.ndarray,
    e1: np.ndarray,
    s2: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """
    Pairwise overlap mask between intervals [s1,e1] and [s2,e2] (inclusive).

    Broadcasts if one side is scalar arrays of shape (1,).
    """
    return (s1 <= e2) & (s2 <= e1)
