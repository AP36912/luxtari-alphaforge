"""Tests for time and information-driven bar builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.bars.information_bars import dollar_bars, tick_bars, volume_bars
from data.bars.time_bars import time_bars
from data.ohlcv_schema import SCHEMA_COLUMNS


def _frame(**kwargs) -> pd.DataFrame:
    base = {
        "timestamp": kwargs.get("timestamp"),
        "open": kwargs["open"],
        "high": kwargs["high"],
        "low": kwargs["low"],
        "close": kwargs["close"],
        "volume": kwargs["volume"],
    }
    return pd.DataFrame(base)


def test_time_bars_aggregates_within_interval_only() -> None:
    ts = pd.to_datetime(
        ["2024-01-01 09:00:00", "2024-01-01 09:00:30", "2024-01-01 09:01:15"]
    )
    df = _frame(
        timestamp=ts,
        open=[10.0, 10.5, 11.0],
        high=[10.5, 10.8, 11.2],
        low=[9.9, 10.4, 10.9],
        close=[10.4, 10.7, 11.1],
        volume=[1.0, 1.0, 1.0],
    )
    out = time_bars(df, "1min")
    assert len(out) == 2
    assert out.iloc[0]["high"] == 10.8
    assert out.iloc[0]["volume"] == 2.0
    assert out.iloc[1]["high"] == 11.2
    assert out.iloc[1]["open"] == 11.0


def test_time_bars_empty() -> None:
    empty = pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    out = time_bars(empty, "1min")
    assert len(out) == 0
    assert list(out.columns) == list(SCHEMA_COLUMNS)


def test_tick_bars_vectorized_groups() -> None:
    n = 10
    ts = pd.date_range("2024-01-01", periods=n, freq="s")
    df = _frame(
        timestamp=ts,
        open=np.arange(n, dtype=float),
        high=np.arange(n, dtype=float) + 0.5,
        low=np.arange(n, dtype=float) - 0.5,
        close=np.arange(n, dtype=float) + 0.25,
        volume=np.ones(n),
    )
    out = tick_bars(df, ticks_per_bar=3)
    assert len(out) == 4
    assert out.iloc[0]["open"] == 0.0
    assert out.iloc[0]["close"] == 2.25
    assert out.iloc[0]["volume"] == 3.0
    assert out.iloc[-1]["volume"] == 1.0


def test_tick_bars_invalid_ticks_per_bar() -> None:
    df = _frame(
        timestamp=pd.to_datetime(["2024-01-01"]),
        open=[1.0],
        high=[1.0],
        low=[1.0],
        close=[1.0],
        volume=[1.0],
    )
    with pytest.raises(ValueError, match="ticks_per_bar"):
        tick_bars(df, ticks_per_bar=0)


def test_volume_bars_threshold_forward_only() -> None:
    df = _frame(
        timestamp=pd.to_datetime(
            ["2024-01-01 10:00:00", "2024-01-01 10:00:01", "2024-01-01 10:00:02"]
        ),
        open=[10.0, 10.1, 10.2],
        high=[10.2, 10.3, 10.4],
        low=[9.9, 10.0, 10.1],
        close=[10.1, 10.2, 10.3],
        volume=[30.0, 25.0, 50.0],
    )
    out = volume_bars(df, volume_threshold=40.0)
    assert len(out) == 2
    assert out.iloc[0]["volume"] == 55.0
    assert out.iloc[0]["high"] == 10.3
    assert out.iloc[1]["volume"] == 50.0


def test_dollar_bars_uses_close_times_volume() -> None:
    # close * vol: 10*10=100, 20*5=100 -> one bar if threshold 150? 100+100=200 >= 150 -> 2 rows in one bar
    df = _frame(
        timestamp=pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:00:01"]),
        open=[10.0, 20.0],
        high=[11.0, 21.0],
        low=[9.0, 19.0],
        close=[10.0, 20.0],
        volume=[10.0, 5.0],
    )
    out = dollar_bars(df, dollar_threshold=150.0)
    assert len(out) == 1
    assert out.iloc[0]["volume"] == 15.0
    assert out.iloc[0]["close"] == 20.0


def test_resample_to_bar_size_alias() -> None:
    from data.bars.resample import resample_to_bar_size

    ts = pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:02:00"])
    df = _frame(
        timestamp=ts,
        open=[1.0, 2.0],
        high=[1.1, 2.1],
        low=[0.9, 1.9],
        close=[1.05, 2.05],
        volume=[1.0, 2.0],
    )
    out = resample_to_bar_size(df, "1h")
    assert len(out) == 1
    assert out["volume"].iloc[0] == 3.0
