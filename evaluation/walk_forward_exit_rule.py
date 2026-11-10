"""
Walk-forward cohort-conditional exit overrides.

These rules run after model prediction and regime entry gates, but before
backtest execution. They are implemented as signal overrides so they work with
the existing signal-driven BacktestEngine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regime.config import RegimeFilterConfig


def _atr_ewm(ohlc: pd.DataFrame, period: int) -> np.ndarray:
    """Causal ATR-like series (Wilder-style EWM on true range)."""
    h = ohlc["high"].to_numpy(dtype=np.float64)
    l = ohlc["low"].to_numpy(dtype=np.float64)
    c = ohlc["close"].to_numpy(dtype=np.float64)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    tr = np.nanmax(
        np.column_stack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]),
        axis=1,
    )
    atr = pd.Series(tr).ewm(
        alpha=1.0 / max(1, int(period)),
        min_periods=max(1, int(period)),
        adjust=False,
    ).mean()
    return atr.to_numpy(dtype=np.float64)


def apply_walk_forward_cohort_exit_rule(
    signals: np.ndarray,
    ohlc: pd.DataFrame,
    vol_states: np.ndarray,
    struct_states: np.ndarray,
    *,
    filters: RegimeFilterConfig,
    fill_basis: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply low_vol|neutral|long defensive exit if enabled in ``filters``.

    Rule:
      For entries where signal bar tags are low_vol + neutral + long,
      if a future bar low reaches entry_price - atr_multiple * ATR(period),
      force signal flat (0) so exit occurs on next fill.
    """
    if not bool(filters.exit_lowvol_neutral_long_defensive_enabled):
        return signals, {"enabled": False, "forced_exits": 0}

    # The current rule is defined for next-bar-open execution.
    if str(fill_basis) != "next_bar_open":
        return signals, {
            "enabled": True,
            "forced_exits": 0,
            "reason": f"unsupported_fill_basis:{fill_basis}",
        }

    sig = np.asarray(signals, dtype=np.float64).copy()
    n = len(sig)
    if n == 0:
        return sig, {"enabled": True, "forced_exits": 0}

    atr_period = int(filters.exit_lowvol_neutral_long_atr_period)
    atr_mult = float(filters.exit_lowvol_neutral_long_atr_multiple)
    atr = _atr_ewm(ohlc, atr_period)
    opens = ohlc["open"].to_numpy(dtype=np.float64)
    lows = ohlc["low"].to_numpy(dtype=np.float64)

    in_target_trade = False
    entry_sig_bar = -1
    entry_price = np.nan
    forced = 0
    target_entries = 0

    for i in range(n):
        if in_target_trade:
            # Trade already closed naturally (flat/opposite) on signal level.
            if sig[i] <= 0.0:
                in_target_trade = False
                continue

            # Evaluate defensive threshold on current bar.
            if np.isfinite(entry_price) and np.isfinite(atr[i]) and atr[i] > 0.0:
                stop_level = entry_price - atr_mult * atr[i]
                if lows[i] <= stop_level:
                    sig[i] = 0.0
                    forced += 1
                    in_target_trade = False
                    continue
        else:
            if sig[i] <= 0.0:
                continue
            if str(vol_states[i]) != "low_vol" or str(struct_states[i]) != "neutral":
                continue

            exec_bar = i + 1
            if exec_bar >= n:
                continue
            target_entries += 1
            entry_sig_bar = i
            entry_price = float(opens[exec_bar])
            in_target_trade = True

    return sig, {
        "enabled": True,
        "forced_exits": int(forced),
        "target_entries_seen": int(target_entries),
        "atr_period": atr_period,
        "atr_multiple": atr_mult,
        "fill_basis": str(fill_basis),
    }
