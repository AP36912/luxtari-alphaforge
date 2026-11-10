"""
OHLCV cleaning: ordering, duplicates, missing values.

All operations are **causal** in the sense that they only use each row's own
fields and sort order; there is no rolling window that looks ahead in time.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from data.ohlcv_schema import enforce_ohlcv_schema, normalize_column_names

MissingVolumePolicy = Literal["zero", "drop_row"]
MissingOhlcPolicy = Literal["drop_row", "ffill"]


def sort_by_timestamp(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    """Stable sort by timestamp (mergesort preserves relative order for ties)."""
    if col not in df.columns:
        raise KeyError(col)
    return df.sort_values(col, kind="mergesort").reset_index(drop=True)


def remove_duplicate_timestamps(
    df: pd.DataFrame,
    *,
    col: str = "timestamp",
) -> pd.DataFrame:
    """
    Collapse rows sharing the same timestamp.

    Rows are stably sorted by ``(timestamp, original_row_order)`` so ``open`` is the first
    row in file order for that timestamp and ``close`` the last—no future rows are used.
    Aggregation: high=max, low=min, volume=sum.
    """
    if col not in df.columns:
        raise KeyError(col)
    work = df.copy().reset_index(drop=True)
    if not work[col].duplicated().any():
        return sort_by_timestamp(work, col=col)

    work["_ord"] = work.index.values
    work = work.sort_values([col, "_ord"], ascending=[True, True], kind="mergesort")
    grouped = work.groupby(col, sort=False)
    out = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).reset_index()
    return sort_by_timestamp(out, col=col)


def handle_missing_values(
    df: pd.DataFrame,
    *,
    volume_policy: MissingVolumePolicy = "zero",
    ohlc_policy: MissingOhlcPolicy = "drop_row",
) -> pd.DataFrame:
    """
    Handle NaNs in OHLCV.

    - OHLC: default drop rows with any NaN in open/high/low/close (safest for backtests).
      Optional `ffill` only fills **within** already-sorted series (still causal if prior row is past).
    - Volume: default fill NaN with 0.0 (missing activity); or drop row.
    """
    work = df.copy()
    if "timestamp" in work.columns:
        work = work.loc[work["timestamp"].notna()].copy()

    ohlc = ["open", "high", "low", "close"]
    for c in ohlc:
        if c not in work.columns:
            raise KeyError(c)

    if ohlc_policy == "drop_row":
        work = work.dropna(subset=ohlc, how="any")
    elif ohlc_policy == "ffill":
        work[ohlc] = work[ohlc].ffill()
        work = work.dropna(subset=ohlc, how="any")
    else:
        raise ValueError(f"Unknown ohlc_policy: {ohlc_policy}")

    if "volume" in work.columns:
        if volume_policy == "zero":
            work["volume"] = work["volume"].fillna(0.0)
        elif volume_policy == "drop_row":
            work = work.dropna(subset=["volume"], how="any")
        else:
            raise ValueError(f"Unknown volume_policy: {volume_policy}")

    return work.reset_index(drop=True)


def preprocess_ohlcv(
    df: pd.DataFrame,
    *,
    normalize_names: bool = True,
    dedupe_timestamps: bool = True,
    volume_policy: MissingVolumePolicy = "zero",
    ohlc_policy: MissingOhlcPolicy = "drop_row",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: normalize headers, enforce schema, sort, dedupe, handle NaNs.

    Returns a DataFrame with columns `SCHEMA_COLUMNS` only.
    """
    work = normalize_column_names(df) if normalize_names else df.copy()
    work = enforce_ohlcv_schema(work, sort=True)
    work = sort_by_timestamp(work)
    if dedupe_timestamps:
        work = remove_duplicate_timestamps(work)
    work = handle_missing_values(work, volume_policy=volume_policy, ohlc_policy=ohlc_policy)
    work = enforce_ohlcv_schema(work, sort=False)
    return work


def identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible no-op typing for callers expecting a DataFrame."""
    return df.copy()
