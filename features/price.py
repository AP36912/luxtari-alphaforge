"""
Price-based features: returns, volatility, ATR, momentum, trend slope.

All series use only **lagged** and **current** observations via ``shift`` and
``rolling`` with ``center=False`` (default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS
from features.config import FeatureConfig
from features.utils import rolling_ols_slope


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing OHLCV columns: {missing}")


def add_price_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Append price / volatility feature columns to a copy of ``df``."""
    _require_ohlcv(df)
    out = df.copy()
    c = out["close"]
    h, l = out["high"], out["low"]

    for h_ in cfg.return_periods:
        if h_ < 1:
            raise ValueError("return_periods must be >= 1")
        name = cfg.prefixed(f"ret_{h_}")
        out[name] = c.pct_change(h_)

    for h_ in cfg.log_return_periods:
        if h_ < 1:
            raise ValueError("log_return_periods must be >= 1")
        name = cfg.prefixed(f"logret_{h_}")
        out[name] = np.log(c / c.shift(h_))

    r1 = c.pct_change(1)
    ann = float(np.sqrt(max(cfg.volatility_annualization_factor, 1e-12)))
    for w in cfg.volatility_windows:
        if w < 2:
            raise ValueError("volatility_windows must be >= 2")
        name = cfg.prefixed(f"rvol_{w}")
        out[name] = r1.rolling(w, min_periods=w).std() * ann

    n_atr = cfg.atr_period
    if n_atr < 1:
        raise ValueError("atr_period must be >= 1")
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out[cfg.prefixed(f"atr_{n_atr}")] = tr.ewm(
        alpha=1.0 / n_atr, min_periods=n_atr, adjust=False
    ).mean()

    for m in cfg.momentum_periods:
        if m < 1:
            raise ValueError("momentum_periods must be >= 1")
        out[cfg.prefixed(f"mom_{m}")] = c - c.shift(m)

    for w in cfg.trend_slope_windows:
        if w < 2:
            raise ValueError("trend_slope_windows must be >= 2")
        out[cfg.prefixed(f"slope_close_{w}")] = rolling_ols_slope(c, w, min_periods=w)

    return out
