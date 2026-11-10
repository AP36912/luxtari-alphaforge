"""
Information-driven bars: tick, volume, and dollar thresholds.

Each bar closes when its cumulative count (ticks), volume, or dollar measure
reaches the threshold, using a **single forward pass** over rows sorted by time.
The bar ``timestamp`` is the last contributing row's time—so the bar is only
known to be complete **at** that event, matching causal ordering for offline
reconstruction (live partial bars are out of scope here).

Dollar measure per row: ``close * volume`` (documented choice for OHLCV rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS, assert_monotonic_timestamps


def _as_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[list(SCHEMA_COLUMNS)].copy()


def _aggregate_by_bar_id(df: pd.DataFrame, bar_id: np.ndarray) -> pd.DataFrame:
    """Vectorized OHLCV aggregation by integer bar id (contiguous groups)."""
    work = _as_ohlcv_frame(df)
    work["_bar_id"] = bar_id
    g = work.groupby("_bar_id", sort=True)
    out = pd.DataFrame(
        {
            "timestamp": g["timestamp"].last(),
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    ).reset_index(drop=True)
    return out


def tick_bars(df: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """
    Every ``ticks_per_bar`` consecutive rows form one bar (same as tick count).

    Rows must already be in time order (one row per tick or event in your feed).
    """
    if ticks_per_bar < 1:
        raise ValueError("ticks_per_bar must be >= 1")
    if df.empty:
        return _as_ohlcv_frame(df)
    work = _as_ohlcv_frame(df)
    assert_monotonic_timestamps(work)
    n = len(work)
    bar_id = np.arange(n, dtype=np.int64) // ticks_per_bar
    return _aggregate_by_bar_id(work, bar_id)


def volume_bars(df: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    """
    New bar when cumulative volume since the last bar start reaches ``volume_threshold``.

    The triggering row is included in the bar; cumulative volume resets after close.
    """
    if volume_threshold <= 0:
        raise ValueError("volume_threshold must be positive")
    if df.empty:
        return _as_ohlcv_frame(df)
    work = _as_ohlcv_frame(df)
    assert_monotonic_timestamps(work)

    o = work["open"].to_numpy(dtype=np.float64)
    h = work["high"].to_numpy(dtype=np.float64)
    lo = work["low"].to_numpy(dtype=np.float64)
    c = work["close"].to_numpy(dtype=np.float64)
    v = work["volume"].to_numpy(dtype=np.float64)

    rows: list[tuple] = []
    i = 0
    n = len(work)
    while i < n:
        vol_acc = 0.0
        o0 = float(o[i])
        hmax = float(h[i])
        lmin = float(lo[i])
        j = i
        while j < n and vol_acc < volume_threshold:
            hmax = max(hmax, float(h[j]))
            lmin = min(lmin, float(lo[j]))
            vol_acc += float(v[j])
            j += 1
        last = j - 1
        ts_last = work["timestamp"].iloc[last]
        rows.append((ts_last, o0, hmax, lmin, float(c[last]), vol_acc))
        i = j

    return pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))


def dollar_bars(df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """
    New bar when cumulative ``close * volume`` reaches ``dollar_threshold``.

    Uses close of each row times its volume as the dollar increment (standard
    OHLCV proxy when tick price is unavailable).
    """
    if dollar_threshold <= 0:
        raise ValueError("dollar_threshold must be positive")
    if df.empty:
        return _as_ohlcv_frame(df)
    work = _as_ohlcv_frame(df)
    assert_monotonic_timestamps(work)

    o = work["open"].to_numpy(dtype=np.float64)
    h = work["high"].to_numpy(dtype=np.float64)
    lo = work["low"].to_numpy(dtype=np.float64)
    c = work["close"].to_numpy(dtype=np.float64)
    v = work["volume"].to_numpy(dtype=np.float64)

    rows: list[tuple] = []
    i = 0
    n = len(work)
    while i < n:
        dollar_acc = 0.0
        o0 = float(o[i])
        hmax = float(h[i])
        lmin = float(lo[i])
        j = i
        while j < n and dollar_acc < dollar_threshold:
            hmax = max(hmax, float(h[j]))
            lmin = min(lmin, float(lo[j]))
            dollar_acc += float(c[j]) * float(v[j])
            j += 1
        last = j - 1
        vol_acc = float(v[i:j].sum())
        ts_last = work["timestamp"].iloc[last]
        rows.append((ts_last, o0, hmax, lmin, float(c[last]), vol_acc))
        i = j

    return pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))
