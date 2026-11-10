"""
Phase 22 — alpha-oriented context features (ASSET / intraday friendly).

All computations are causal (rolling/shift with ``center=False``). Requires
prior blocks: :mod:`features.price` (ATR), :mod:`features.technical` (BB, MAs).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS
from features.config import FeatureConfig
from features.utils import rolling_ols_slope
from regime.config import SessionCalendarConfig
from regime.session_context import compute_session_tags

logger = logging.getLogger(__name__)

# Prefixes for columns added by this module (for logging / grouping).
DISCOVERY_FEATURE_PREFIXES: tuple[str, ...] = (
    "mom_ret_",
    "sess_oh_",
    "volreg_",
    "pos_",
    "trend_",
)


def discovery_feature_names(columns: list[str]) -> list[str]:
    """Return discovery column names among ``columns`` (stable order)."""
    out: list[str] = []
    for c in columns:
        if any(c.startswith(p) for p in DISCOVERY_FEATURE_PREFIXES):
            out.append(c)
    return sorted(out)


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing OHLCV columns: {missing}")


def _wilder_ewm(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder-style ADX(period); causal."""
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_c).abs(),
            (low - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    sm_tr = _wilder_ewm(tr, period)
    sm_plus = _wilder_ewm(plus_dm, period)
    sm_minus = _wilder_ewm(minus_dm, period)
    denom = sm_tr.replace(0.0, np.nan)
    plus_di = 100.0 * (sm_plus / denom)
    minus_di = 100.0 * (sm_minus / denom)
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder_ewm(dx, period)
    return adx


def _rolling_percentile_rank_last(x: pd.Series, window: int) -> pd.Series:
    """Fraction of window values <= current (inclusive); causal."""

    def _rank_last(arr: np.ndarray) -> float:
        if arr.size < window or not np.all(np.isfinite(arr)):
            return np.nan
        last = arr[-1]
        return float((arr <= last).mean())

    return x.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def add_discovery_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Session one-hots, return moments, ATR regime flags, band distances, ADX, direction stats.

    No-op when ``cfg.discovery_enabled`` is False.
    """
    _require_ohlcv(df)
    if not cfg.discovery_enabled:
        return df

    out = df.copy()
    close = out["close"]
    high, low = out["high"], out["low"]

    w_ret = cfg.discovery_ret_stats_window
    if w_ret < 2:
        raise ValueError("discovery_ret_stats_window must be >= 2")
    r1 = close.pct_change(1)
    r_mean = r1.rolling(w_ret, min_periods=w_ret).mean()
    r_std = r1.rolling(w_ret, min_periods=w_ret).std(ddof=0)
    out[cfg.prefixed(f"mom_ret_z_{w_ret}")] = (r1 - r_mean) / r_std.replace(0.0, np.nan)
    out[cfg.prefixed(f"mom_ret_mean_{w_ret}")] = r_mean
    out[cfg.prefixed(f"mom_ret_std_{w_ret}")] = r_std

    sess_cal = SessionCalendarConfig(
        session_timezone=cfg.discovery_session_timezone,
        assume_timestamp_utc_if_naive=cfg.discovery_session_assume_utc_naive,
    )
    tags = compute_session_tags(out, sess_cal)
    liquid = frozenset({"asia", "london", "new_york", "london_ny_overlap"})
    for name in ("asia", "london", "new_york", "london_ny_overlap"):
        out[cfg.prefixed(f"sess_oh_{name}")] = tags.eq(name).astype("float64")
    out[cfg.prefixed("sess_oh_other")] = (~tags.isin(liquid)).astype("float64")

    w_atr = cfg.discovery_atr_pct_window
    if w_atr < 5:
        raise ValueError("discovery_atr_pct_window must be >= 5")
    p_atr = cfg.atr_period
    atr_col = cfg.prefixed(f"atr_{p_atr}")
    if atr_col not in out.columns:
        raise KeyError(f"discovery requires {atr_col} from price features")
    atr = out[atr_col]
    out[cfg.prefixed(f"volreg_atr_pct_{w_atr}")] = _rolling_percentile_rank_last(atr, w_atr)
    med = atr.rolling(w_atr, min_periods=w_atr).median()
    q25 = atr.rolling(w_atr, min_periods=w_atr).quantile(0.25)
    out[cfg.prefixed("volreg_high_atr")] = (atr > med).astype("float64")
    out[cfg.prefixed("volreg_low_atr")] = (atr < q25).astype("float64")

    bw = cfg.bollinger_window
    mid_c = cfg.prefixed(f"bb_mid_{bw}")
    up_c = cfg.prefixed(f"bb_upper_{bw}")
    lo_c = cfg.prefixed(f"bb_lower_{bw}")
    if mid_c not in out.columns or up_c not in out.columns or lo_c not in out.columns:
        raise KeyError("discovery requires Bollinger columns from technical features")

    span = cfg.discovery_position_ema_span
    ema_c = cfg.prefixed(f"ema_{span}")
    if ema_c in out.columns:
        ema_ref = out[ema_c]
    else:
        ema_ref = close.ewm(span=span, min_periods=span, adjust=False).mean()

    atr_safe = atr.replace(0.0, np.nan)
    out[cfg.prefixed(f"pos_dist_ema{span}_atr")] = (close - ema_ref) / atr_safe
    out[cfg.prefixed(f"pos_dist_bb_mid_{bw}_atr")] = (close - out[mid_c]) / atr_safe
    out[cfg.prefixed(f"pos_dist_bb_upper_{bw}_atr")] = (out[up_c] - close) / atr_safe
    out[cfg.prefixed(f"pos_dist_bb_lower_{bw}_atr")] = (close - out[lo_c]) / atr_safe

    adx_p = cfg.discovery_adx_period
    if adx_p < 2:
        raise ValueError("discovery_adx_period must be >= 2")
    out[cfg.prefixed(f"trend_adx_{adx_p}")] = _adx(high, low, close, adx_p)

    w_slope = cfg.discovery_trend_slope_window
    if w_slope >= 2:
        out[cfg.prefixed(f"trend_slope_close_{w_slope}")] = rolling_ols_slope(
            close, w_slope, min_periods=w_slope
        )

    w_dir = cfg.discovery_dir_consistency_window
    if w_dir >= 2:
        up = (close.diff() > 0.0).astype("float64")
        out[cfg.prefixed(f"trend_close_up_ratio_{w_dir}")] = up.rolling(
            w_dir, min_periods=w_dir
        ).mean()

    added = discovery_feature_names([c for c in out.columns if c not in df.columns])
    logger.debug("discovery_features_added count=%s names=%s", len(added), added)
    return out
