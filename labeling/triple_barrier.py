"""
Triple-barrier labels (profit / stop / time cap).

For each decision index ``t`` the path is evaluated on **strictly future** bars
``t+1, t+2, ...`` only. Intrabar sequencing is **conservative** for the active
side: for a long, stop-style touches (low / gap down through stop) are checked
before take-profit touches (high / gap up through profit) on the same bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS
from labeling.config import TripleBarrierConfig


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing OHLCV columns: {missing}")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)


def _volatility_series(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.Series:
    """Causal volatility at each bar (known at bar close)."""
    close = df["close"]
    if cfg.volatility_scale == "returns_std":
        r = close.pct_change()
        return r.rolling(cfg.volatility_window, min_periods=cfg.volatility_window).std()
    if cfg.volatility_scale == "atr":
        tr = _true_range(df["high"], df["low"], close)
        atr = tr.ewm(
            alpha=1.0 / cfg.atr_period, min_periods=cfg.atr_period, adjust=False
        ).mean()
        return atr / close.replace(0.0, np.nan)
    raise ValueError(f"Unknown volatility_scale: {cfg.volatility_scale!r}")


def _barrier_widths(vol: float, cfg: TripleBarrierConfig) -> tuple[float, float]:
    """Return (profit_width, stop_width) as **fractional** distance from entry."""
    if cfg.use_volatility_scaling:
        if not np.isfinite(vol) or vol < 0:
            return np.nan, np.nan
        return cfg.profit_vol_multiple * float(vol), cfg.stop_vol_multiple * float(vol)
    return cfg.profit_pct, cfg.stop_pct


def _scan_long(
    j_start: int,
    j_end_exclusive: int,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[int | None, str | None, int | None]:
    for j in range(j_start, j_end_exclusive):
        oj, hj, lj = float(o[j]), float(h[j]), float(l[j])
        if oj >= upper:
            return 1, "take_profit", j
        if oj <= lower:
            return -1, "stop_loss", j
        if lj <= lower:
            return -1, "stop_loss", j
        if hj >= upper:
            return 1, "take_profit", j
    return None, None, None


def _scan_short(
    j_start: int,
    j_end_exclusive: int,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    profit_level: float,
    stop_level: float,
) -> tuple[int | None, str | None, int | None]:
    for j in range(j_start, j_end_exclusive):
        oj, hj, lj = float(o[j]), float(h[j]), float(l[j])
        if oj <= profit_level:
            return 1, "take_profit", j
        if oj >= stop_level:
            return -1, "stop_loss", j
        if hj >= stop_level:
            return -1, "stop_loss", j
        if lj <= profit_level:
            return 1, "take_profit", j
    return None, None, None


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    cfg: TripleBarrierConfig,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels aligned to each row's decision time.

    The label at row ``t`` uses entry from that row and OHLC paths on rows
    ``t+1 ..`` only (aligned with features at ``t`` that use information in
    ``..t``).
    """
    _require_ohlcv(df)
    work = df.reset_index(drop=True)
    n = len(work)
    o = work["open"].to_numpy(dtype=np.float64)
    h = work["high"].to_numpy(dtype=np.float64)
    l = work["low"].to_numpy(dtype=np.float64)
    close = work["close"].to_numpy(dtype=np.float64)
    open_ = work["open"].to_numpy(dtype=np.float64)
    entry_series = close if cfg.entry_price_column == "close" else open_
    vol_series = _volatility_series(work, cfg) if cfg.use_volatility_scaling else None

    labels = np.full(n, np.nan, dtype=np.float64)
    events: list[str] = []
    bars_to = np.full(n, np.nan, dtype=np.float64)

    side = cfg.position_side

    for t in range(n):
        entry = float(entry_series[t])
        if cfg.use_volatility_scaling:
            assert vol_series is not None
            pw, sw = _barrier_widths(float(vol_series.iloc[t]), cfg)
        else:
            pw, sw = cfg.profit_pct, cfg.stop_pct

        if cfg.max_bars < 0:
            raise ValueError("max_bars must be non-negative")
        if cfg.max_bars == 0:
            labels[t] = 0.0
            events.append("vertical")
            bars_to[t] = 0.0
            continue

        j_start = t + 1
        j_end_exclusive = min(t + 1 + cfg.max_bars, n)

        if j_start >= n:
            labels[t] = np.nan
            events.append("insufficient")
            bars_to[t] = np.nan
            continue

        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(pw) or not np.isfinite(sw):
            labels[t] = np.nan
            events.append("insufficient")
            bars_to[t] = np.nan
            continue

        if side == "long":
            upper = entry * (1.0 + pw)
            lower = max(entry * (1.0 - sw), 1e-12)
            hit, ev, j_hit = _scan_long(j_start, j_end_exclusive, o, h, l, upper, lower)
        else:
            profit_level = entry * (1.0 - pw)
            stop_level = entry * (1.0 + sw)
            hit, ev, j_hit = _scan_short(
                j_start, j_end_exclusive, o, h, l, profit_level, stop_level
            )

        if hit is not None and j_hit is not None:
            labels[t] = float(hit)
            events.append(ev if ev else "take_profit")
            bars_to[t] = float(j_hit - t)
            continue

        if j_end_exclusive == t + 1 + cfg.max_bars:
            labels[t] = 0.0
            events.append("vertical")
            bars_to[t] = float(cfg.max_bars)
        else:
            labels[t] = np.nan
            events.append("insufficient")
            bars_to[t] = np.nan

    return pd.DataFrame(
        {
            "triple_barrier_label": labels,
            "triple_barrier_event": events,
            "bars_to_event": bars_to,
        }
    )


def merge_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Concatenate OHLCV (or feature) frame with label columns, row-aligned."""
    if len(df) != len(labels):
        raise ValueError("df and labels must have the same number of rows")
    return pd.concat([df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
