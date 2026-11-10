"""Deterministic synthetic OHLCV generation for examples and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(*, rows: int = 750, seed: int = 42, start: float = 100.0,
                             timeframe: str = "1h") -> pd.DataFrame:
    """Generate a seeded geometric random walk with internally consistent bars."""
    if rows < 50:
        raise ValueError("rows must be at least 50")
    if start <= 0:
        raise ValueError("start must be positive")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00005, 0.006, rows)
    close = start * np.exp(np.cumsum(returns))
    open_ = np.r_[start, close[:-1]]
    span = np.abs(rng.normal(0.002, 0.001, rows))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    volume = rng.lognormal(mean=7.0, sigma=0.3, size=rows)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=rows, freq=timeframe),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })
