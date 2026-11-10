"""
Canonical OHLCV schema for the data pipeline.

All loaders and bar generators target the same column set so downstream code
never infers semantics per vendor. Timestamps are **event times** for each row
( bar end or tick time ); bar builders must not shift labels forward in time
relative to the last contributing row (see `data/bars/` docstrings).
"""

from __future__ import annotations

import re

import pandas as pd

SCHEMA_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")

# Recognized aliases -> canonical name (after lowercasing and stripping).
_COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "ticktime": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
    "vol": "volume",
    "tickvol": "volume",
    "tick_volume": "volume",
    "real_volume": "volume",
    "tickvolume": "volume",
}


def _sanitize_header(name: str) -> str:
    s = str(name).strip()
    s = s.strip("<>").strip()
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map vendor headers to canonical names where recognized.

    Unmapped columns are kept sanitized (lowercase snake_case) for debugging;
    `enforce_ohlcv_schema` will drop extras.
    """
    out = df.copy()
    new_cols: list[str] = []
    for c in out.columns:
        key = _sanitize_header(c)
        new_cols.append(_COLUMN_ALIASES.get(key, key))
    out.columns = new_cols
    return out


def enforce_ohlcv_schema(
    df: pd.DataFrame,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Ensure required columns exist, dtypes are numeric/time-safe, and order is fixed.

    Does not dedupe or impute; use `preprocessing.clean.preprocess_ohlcv` for full hygiene.
    """
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV schema missing columns: {missing}")

    out = df[list(SCHEMA_COLUMNS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if sort:
        out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out


def ohlcv_to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper: column `timestamp` -> DatetimeIndex (sorted)."""
    out = enforce_ohlcv_schema(df, sort=True)
    out = out.set_index("timestamp")
    out.index.name = "timestamp"
    return out


def assert_monotonic_timestamps(df: pd.DataFrame) -> None:
    """Raise if timestamps are not non-decreasing (mergesort-stable duplicates allowed)."""
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending for causal bar rules")
