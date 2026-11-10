"""
Fixed calendar time bars (OHLCV) via pandas resampling.

Uses ``closed='left'`` and ``label='left'`` so each interval ``[t, t + freq)``
only aggregates rows with timestamps ``>= t`` and ``< t + freq``—no future
rows enter an earlier bar's aggregates. The bar's ``timestamp`` is the **left**
edge of that interval (interval start).
"""

from __future__ import annotations

import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS, assert_monotonic_timestamps, ohlcv_to_datetime_index


def time_bars(
    df: pd.DataFrame,
    freq: str,
    *,
    closed: str = "left",
    label: str = "left",
) -> pd.DataFrame:
    """
    Aggregate ticks or lower-timeframe bars to ``freq`` (pandas offset string).

    Args:
        df: Canonical OHLCV (sorted ascending by ``timestamp``).
        freq: Pandas frequency (e.g. ``"1min"``, ``"5min"``, ``"1h"``).
        closed: Which side of each bin is closed (default ``left``).
        label: Which bin edge is used as the label / returned ``timestamp``.
    """
    if df.empty:
        return pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    work = df[list(SCHEMA_COLUMNS)].copy()
    assert_monotonic_timestamps(work)
    idxed = ohlcv_to_datetime_index(work)
    agg = idxed.resample(freq, closed=closed, label=label).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"], how="any")
    out = agg.reset_index()
    out.columns = list(SCHEMA_COLUMNS)
    return out.reset_index(drop=True)
