"""
Shared numerical helpers for causal rolling features.

Rolling windows use pandas defaults: ``center=False`` (past and current bar only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols_slope(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """
    Rolling OLS slope of ``series`` vs integer time 0..window-1 within each window.

    At index ``t``, uses observations ``[t-window+1, ..., t]`` only (causal).
    """
    if min_periods is None:
        min_periods = window

    def slope_window(y: np.ndarray) -> float:
        w = y.shape[0]
        if w < 2 or not np.all(np.isfinite(y)):
            return np.nan
        x = np.arange(w, dtype=np.float64)
        var_x = np.var(x)
        if var_x < 1e-18:
            return np.nan
        cov_xy = np.mean((x - x.mean()) * (y - y.mean()))
        return float(cov_xy / var_x)

    return series.rolling(window, min_periods=min_periods).apply(slope_window, raw=True)
