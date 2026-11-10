"""
Rolling statistical features: z-score, variance, skewness, kurtosis.

Uses pandas rolling with ``center=False`` so statistics at time ``t`` depend
only on observations with index ``<= t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS
from features.config import FeatureConfig


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing OHLCV columns: {missing}")


def add_statistical_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Append rolling distribution shape features."""
    _require_ohlcv(df)
    out = df.copy()

    for col in cfg.zscore_columns:
        if col not in out.columns:
            raise KeyError(f"zscore_columns references missing column: {col}")
        s = out[col]
        for w in cfg.zscore_windows:
            if w < 2:
                raise ValueError("zscore_windows must be >= 2")
            mean = s.rolling(w, min_periods=w).mean()
            std = s.rolling(w, min_periods=w).std(ddof=0)
            out[cfg.prefixed(f"zscore_{col}_{w}")] = (s - mean) / std.replace(0.0, np.nan)

    for w in cfg.rolling_variance_windows:
        if w < 2:
            raise ValueError("rolling_variance_windows must be >= 2")
        out[cfg.prefixed(f"rollvar_close_{w}")] = (
            out["close"].rolling(w, min_periods=w).var(ddof=0)
        )

    for w in cfg.skew_windows:
        if w < 3:
            raise ValueError("skew_windows must be >= 3 for meaningful skew")
        out[cfg.prefixed(f"rollskew_close_{w}")] = (
            out["close"].rolling(w, min_periods=w).skew()
        )

    for w in cfg.kurtosis_windows:
        if w < 4:
            raise ValueError("kurtosis_windows must be >= 4 for meaningful kurtosis")
        out[cfg.prefixed(f"rollkurt_close_{w}")] = (
            out["close"].rolling(w, min_periods=w).kurt()
        )

    return out
