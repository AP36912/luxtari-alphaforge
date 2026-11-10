"""
Compatibility wrapper for time-based resampling.

Prefer importing ``time_bars`` from ``data.bars.time_bars`` in new code.
"""

from __future__ import annotations

import pandas as pd

from data.bars.time_bars import time_bars


def resample_to_bar_size(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Delegate to :func:`time_bars` with ``freq=rule``."""
    return time_bars(frame, freq=rule)
