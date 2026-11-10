"""Tests for preprocessing and schema enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from data.ohlcv_schema import SCHEMA_COLUMNS, enforce_ohlcv_schema, normalize_column_names
from data.preprocessing.clean import (
    handle_missing_values,
    preprocess_ohlcv,
    remove_duplicate_timestamps,
    sort_by_timestamp,
)


def test_normalize_column_names_maps_aliases() -> None:
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Vol": [10.0],
            "Datetime": ["2024-01-01 10:00:00"],
        }
    )
    out = normalize_column_names(df)
    assert "open" in out.columns
    assert "volume" in out.columns
    assert "timestamp" in out.columns


def test_preprocess_sorts_and_dedupes_timestamps() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 10:01", "2024-01-01 10:00", "2024-01-01 10:00"],
            "open": [2.0, 1.0, 1.1],
            "high": [2.5, 1.5, 1.4],
            "low": [1.9, 0.9, 1.0],
            "close": [2.2, 1.2, 1.15],
            "volume": [10.0, 5.0, 3.0],
        }
    )
    out = preprocess_ohlcv(raw, normalize_names=False)
    assert len(out) == 2
    assert out.iloc[0]["timestamp"] < out.iloc[1]["timestamp"]
    # duplicate 10:00: open first (1.0), high max(1.5,1.4), low min(0.9,1.0), close last (1.15), vol sum 8
    row0 = out.loc[out["timestamp"] == pd.Timestamp("2024-01-01 10:00:00")].iloc[0]
    assert row0["open"] == 1.0
    assert row0["high"] == 1.5
    assert row0["low"] == 0.9
    assert row0["close"] == 1.15
    assert row0["volume"] == 8.0


def test_handle_missing_drops_bad_ohlc() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [1.0, float("nan")],
            "high": [2.0, 2.0],
            "low": [0.5, 1.0],
            "close": [1.5, 1.8],
            "volume": [10.0, 5.0],
        }
    )
    out = handle_missing_values(df, ohlc_policy="drop_row")
    assert len(out) == 1


def test_handle_missing_volume_zero_fill() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [float("nan")],
        }
    )
    out = handle_missing_values(df, volume_policy="zero")
    assert out.iloc[0]["volume"] == 0.0


def test_enforce_schema_raises_on_missing_column() -> None:
    df = pd.DataFrame({"open": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        enforce_ohlcv_schema(df)


def test_sort_by_timestamp_stable() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 10:01"), pd.Timestamp("2024-01-01 10:00")],
            "x": [2, 1],
        }
    )
    out = sort_by_timestamp(df)
    assert list(out["x"]) == [1, 2]
