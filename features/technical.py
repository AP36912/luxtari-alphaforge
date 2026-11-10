"""
Technical indicators: RSI, MACD, Bollinger bands, moving averages, MA slopes.

Implemented with standard textbook definitions; all operations are causal
(past and current bars only).
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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _moving_average(close: pd.Series, window: int, kind: str) -> pd.Series:
    kind_l = kind.lower()
    if kind_l == "sma":
        return close.rolling(window, min_periods=window).mean()
    if kind_l == "ema":
        return close.ewm(span=window, min_periods=window, adjust=False).mean()
    raise ValueError(f"Unknown ma type: {kind!r} (use 'sma' or 'ema')")


def add_technical_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Append technical indicator columns."""
    _require_ohlcv(df)
    out = df.copy()
    close = out["close"]
    p = cfg.rsi_period
    if p < 2:
        raise ValueError("rsi_period must be >= 2")
    out[cfg.prefixed(f"rsi_{p}")] = _rsi(close, p)

    fast, slow, sig = cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    if fast < 1 or slow < 1 or sig < 1 or fast >= slow:
        raise ValueError("Invalid MACD periods (need fast < slow, all positive)")
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=sig, min_periods=sig, adjust=False).mean()
    out[cfg.prefixed("macd")] = macd_line
    out[cfg.prefixed("macd_signal")] = signal
    out[cfg.prefixed("macd_hist")] = macd_line - signal

    bw = cfg.bollinger_window
    k = cfg.bollinger_num_std
    if bw < 2:
        raise ValueError("bollinger_window must be >= 2")
    mid = close.rolling(bw, min_periods=bw).mean()
    std = close.rolling(bw, min_periods=bw).std(ddof=0)
    out[cfg.prefixed(f"bb_mid_{bw}")] = mid
    out[cfg.prefixed(f"bb_upper_{bw}")] = mid + k * std
    out[cfg.prefixed(f"bb_lower_{bw}")] = mid - k * std

    for w in cfg.ma_windows:
        if w < 1:
            raise ValueError("ma_windows must be >= 1")
        for mt in cfg.ma_types:
            col = cfg.prefixed(f"{mt.lower()}_{w}")
            out[col] = _moving_average(close, w, mt)

    for w in cfg.ma_slope_windows:
        if w < 2:
            raise ValueError("ma_slope_windows must be >= 2")
        sma_w = close.rolling(w, min_periods=w).mean()
        out[cfg.prefixed(f"sma_slope_{w}")] = rolling_ols_slope(sma_w, w, min_periods=w)

    return out
