"""
Rolling volatility regime labels: high / mid / low vs expanding distribution (causal quantiles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import RegimeDetectionParams


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of simple returns; causal (uses data up to and including t)."""
    r = close.pct_change()
    return r.rolling(window, min_periods=max(3, window // 3)).std()


def volatility_state_series(close: pd.Series, p: RegimeDetectionParams) -> pd.Series:
    """
    Per-bar label: ``high_vol``, ``low_vol``, or ``mid_vol``.

    Compares short-window vol at ``t`` to rolling quantiles of past short vol (shifted by 1).
    """
    w_s = max(3, p.vol_short_window)
    w_l = max(w_s + 1, p.vol_long_window)
    w_q = max(w_l + 1, p.vol_quantile_window)

    v = realized_volatility(close, w_s)
    v_past = v.shift(1)
    q_hi = v_past.rolling(w_q, min_periods=max(10, w_q // 4)).quantile(p.vol_high_quantile)
    q_lo = v_past.rolling(w_q, min_periods=max(10, w_q // 4)).quantile(p.vol_low_quantile)

    out = pd.Series("mid_vol", index=close.index, dtype="object")
    hi = v > q_hi
    lo = v < q_lo
    out = out.mask(hi, "high_vol")
    out = out.mask(lo, "low_vol")
    out = out.mask(v.isna() | q_hi.isna() | q_lo.isna(), "mid_vol")
    return out


def vol_state_to_id(series: pd.Series) -> np.ndarray:
    m = {"low_vol": 0, "mid_vol": 1, "high_vol": 2}
    return np.array([m.get(str(x), 1) for x in series], dtype=np.int64)
