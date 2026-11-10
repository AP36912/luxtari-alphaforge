"""
Rule-based market structure regimes: trending, range, expansion, compression (causal OHLC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import RegimeDetectionParams
from regime.volatility_regime import realized_volatility, volatility_state_series


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def _rolling_slope_close(close: pd.Series, window: int) -> pd.Series:
    """OLS slope of close vs 0..window-1 within each trailing window."""
    y = close.astype(np.float64).to_numpy()
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    w = max(3, window)
    x = np.arange(w, dtype=np.float64)
    sx = x.sum()
    sxx = float((x * x).sum())
    denom = w * sxx - sx * sx
    if abs(denom) < 1e-15:
        return pd.Series(out, index=close.index)
    for i in range(w - 1, n):
        seg = y[i - w + 1 : i + 1]
        sy = float(seg.sum())
        sxy = float(np.dot(x, seg))
        out[i] = (w * sxy - sx * sy) / denom
    return pd.Series(out, index=close.index)


STRUCTURE_ORDER = (
    "breakout_expand",
    "compressed",
    "trending_up",
    "trending_down",
    "range_bound",
    "neutral",
)


def structure_to_id(series: pd.Series) -> np.ndarray:
    m = {name: i for i, name in enumerate(STRUCTURE_ORDER)}
    unk = len(m)
    return np.array([m.get(str(x), unk) for x in series], dtype=np.int64)


def detect_structure_series(
    df: pd.DataFrame,
    p: RegimeDetectionParams,
) -> pd.Series:
    """
    Classify each bar into a structure regime using ATR ratios, range compression, slope, momentum.

    Priority: ``breakout_expand`` → ``compressed`` → ``trending_*`` → ``range_bound`` → ``neutral``.
    """
    required = ("open", "high", "low", "close")
    for c in required:
        if c not in df.columns:
            raise KeyError(f"detect_structure_series missing column {c!r}")
    high, low, close = df["high"], df["low"], df["close"]

    atr_s = _atr(high, low, close, max(2, p.atr_short_period))
    atr_l = _atr(high, low, close, max(3, p.atr_long_period))
    atr_ratio = atr_s / atr_l.replace(0.0, np.nan)

    rw = max(3, p.range_window)
    hi_w = high.rolling(rw, min_periods=rw // 2).max()
    lo_w = low.rolling(rw, min_periods=rw // 2).min()
    bar_range = hi_w - lo_w

    slope = _rolling_slope_close(close, max(3, p.slope_window))
    atr_ref = atr_s.replace(0.0, np.nan)
    slope_norm = slope / atr_ref

    mom = close.pct_change(max(1, p.momentum_window))
    mom_thr = p.momentum_confirm_ratio

    out = pd.Series("neutral", index=df.index, dtype="object")

    expand = atr_ratio > p.atr_expand_ratio
    compress = (bar_range < atr_s * p.compression_range_atr_ratio) & atr_s.notna()

    trend_up = (slope_norm > p.slope_trending_atr_ratio) & (mom > mom_thr)
    trend_dn = (slope_norm < -p.slope_trending_atr_ratio) & (mom < -mom_thr)
    range_b = slope_norm.abs() < p.slope_range_atr_ratio

    out = out.mask(expand, "breakout_expand")
    out = out.mask(compress & ~expand, "compressed")
    out = out.mask(trend_up & ~expand & ~compress, "trending_up")
    out = out.mask(trend_dn & ~expand & ~compress & ~trend_up, "trending_down")
    out = out.mask(range_b & ~expand & ~compress & ~trend_up & ~trend_dn, "range_bound")
    return out


def compute_regime_feature_matrix(df: pd.DataFrame, p: RegimeDetectionParams) -> np.ndarray:
    """
    Low-dimensional causal vector per row for offline clustering (``cluster_offline``).

    Columns: ``log_vol_short``, ``atr_ratio``, ``slope_norm``, ``range_atr_ratio``.
    """
    close = df["close"].astype(np.float64)
    high, low = df["high"].astype(np.float64), df["low"].astype(np.float64)
    v = realized_volatility(close, max(3, p.vol_short_window))
    log_v = np.log(v.clip(lower=1e-12))
    atr_s = _atr(high, low, close, max(2, p.atr_short_period))
    atr_l = _atr(high, low, close, max(3, p.atr_long_period))
    atr_ratio = (atr_s / atr_l.replace(0.0, np.nan)).astype(np.float64)
    slope = _rolling_slope_close(close, max(3, p.slope_window))
    slope_norm = (slope / atr_s.replace(0.0, np.nan)).astype(np.float64)
    rw = max(3, p.range_window)
    hi_w = high.rolling(rw, min_periods=rw // 2).max()
    lo_w = low.rolling(rw, min_periods=rw // 2).min()
    bar_range = hi_w - lo_w
    range_atr = (bar_range / atr_s.replace(0.0, np.nan)).astype(np.float64)
    mat = np.column_stack(
        [
            log_v.to_numpy(dtype=np.float64),
            atr_ratio.to_numpy(dtype=np.float64),
            slope_norm.to_numpy(dtype=np.float64),
            range_atr.to_numpy(dtype=np.float64),
        ]
    )
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


def build_rule_based_regime_frame(df: pd.DataFrame, p: RegimeDetectionParams) -> pd.DataFrame:
    """Return columns ``regime_vol_state``, ``regime_structure`` (strings)."""
    vol_st = volatility_state_series(df["close"], p)
    struct = detect_structure_series(df, p)
    out = pd.DataFrame(
        {
            "regime_vol_state": vol_st.values,
            "regime_structure": struct.values,
        },
        index=df.index,
    )
    return out
