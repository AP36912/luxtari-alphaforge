"""
Bar-level microstructure **proxies** (not true order-book data).

These are coarse signed-volume and volatility-ratio heuristics suitable when
only OHLCV is available. They remain causal (rolling over past and current bars).
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


def add_microstructure_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Append microstructure proxy columns."""
    _require_ohlcv(df)
    out = df.copy()
    close, open_, vol = out["close"], out["open"], out["volume"]

    # Bar "tick" direction proxy: sign of close change (zero when flat)
    tick_dir = np.sign(close.diff().fillna(0.0))
    w_dir = cfg.tick_direction_window
    if w_dir < 1:
        raise ValueError("tick_direction_window must be >= 1")
    out[cfg.prefixed(f"tick_dir_sum_{w_dir}")] = tick_dir.rolling(
        w_dir, min_periods=w_dir
    ).sum()

    # Signed volume: allocate volume to "buy" side if close >= open else sell
    signed_vol = np.where(close >= open_, vol, -vol)
    signed_vol = pd.Series(signed_vol, index=out.index, dtype=np.float64)
    w_imb = cfg.volume_imbalance_window
    if w_imb < 1:
        raise ValueError("volume_imbalance_window must be >= 1")
    out[cfg.prefixed(f"vol_imbalance_{w_imb}")] = signed_vol.rolling(
        w_imb, min_periods=w_imb
    ).sum()

    r1 = close.pct_change(1)
    s_short = cfg.volatility_cluster_short
    s_long = cfg.volatility_cluster_long
    if s_short < 2 or s_long < 2 or s_short >= s_long:
        raise ValueError(
            "volatility_cluster_short/long must be >=2 and short < long"
        )
    vol_s = r1.rolling(s_short, min_periods=s_short).std()
    vol_l = r1.rolling(s_long, min_periods=s_long).std()
    out[cfg.prefixed(f"vol_cluster_{s_short}_{s_long}")] = vol_s / vol_l.replace(
        0.0, np.nan
    )

    return out
