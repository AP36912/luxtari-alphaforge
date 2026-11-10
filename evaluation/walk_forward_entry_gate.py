"""
Walk-forward entry-only regime blocks (flat → open), aligned with paper/sim execution safety policy.

Uses the same bar/signal indexing as :class:`backtest.engine.BacktestEngine` for
``next_bar_open`` and ``bar_close`` fill bases.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtest.config import PositionSizing


def apply_walk_forward_entry_regime_blocks(
    signals: np.ndarray,
    vol_states: np.ndarray | None,
    structure_states: np.ndarray | None,
    *,
    fill_basis: str,
    sizing: PositionSizing,
    block_high_vol: bool,
    block_trending_down: bool,
    block_lowvol_rb_london: bool = False,
    session_states: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Zero signals only when opening from flat and a regime block applies.

    ``vol_states[k]`` / ``structure_states[k]`` / ``session_states[k]`` align
    with signal bar ``k`` (same convention as merged test rows).
    """
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    n = sig.shape[0]
    out = sig.copy()

    if fill_basis not in ("next_bar_open", "bar_close"):
        raise ValueError(f"unsupported fill_basis: {fill_basis!r}")

    fs = float(sizing.fixed_shares) if sizing.mode == "fixed_shares" else 1.0
    tol = 1e-12

    pos = 0.0
    blocked_hv = 0
    blocked_td = 0
    blocked_lrl = 0
    n_entry_opportunities = 0

    for k in range(n):
        raw = float(out[k])
        flat = abs(pos) < tol
        opening = flat and abs(raw) > tol
        if opening:
            n_entry_opportunities += 1
            did = False
            if block_high_vol and vol_states is not None and str(vol_states[k]) == "high_vol":
                out[k] = 0.0
                raw = 0.0
                blocked_hv += 1
                did = True
            if (
                not did
                and block_trending_down
                and structure_states is not None
                and str(structure_states[k]) == "trending_down"
            ):
                out[k] = 0.0
                raw = 0.0
                blocked_td += 1
                did = True
            if (
                not did
                and block_lowvol_rb_london
                and vol_states is not None
                and structure_states is not None
                and session_states is not None
                and str(vol_states[k]) == "low_vol"
                and str(structure_states[k]) == "range_bound"
                and str(session_states[k]) == "london"
            ):
                out[k] = 0.0
                raw = 0.0
                blocked_lrl += 1

        if sizing.mode == "fixed_shares":
            pos = float(raw * fs) if abs(raw) > tol else 0.0
        else:
            pos = float(np.sign(raw) * fs) if abs(raw) > tol else 0.0

    pct_hv = 100.0 * blocked_hv / n_entry_opportunities if n_entry_opportunities else 0.0
    diag: dict[str, Any] = {
        "entries_blocked_high_vol": int(blocked_hv),
        "entries_blocked_trending_down": int(blocked_td),
        "entries_blocked_lowvol_rb_london": int(blocked_lrl),
        "n_flat_to_open_opportunities": int(n_entry_opportunities),
        "pct_entry_opportunities_blocked_high_vol": round(pct_hv, 6),
    }
    return out, diag


def trades_by_entry_vol_regime(
    position_series: np.ndarray,
    vol_states: np.ndarray,
) -> dict[str, int]:
    """
    Count new positions (flat → non-flat) by ``vol_states`` at the signal index
    that caused the fill (``next_bar_open``: transition at bar i uses signal i-1).
    """
    pos = np.asarray(position_series, dtype=np.float64).reshape(-1)
    vol = np.asarray(vol_states).reshape(-1)
    n = len(pos)
    if len(vol) != n:
        raise ValueError("position_series and vol_states length mismatch")
    buckets: dict[str, int] = {}
    for i in range(1, n):
        if abs(pos[i - 1]) < 1e-12 and abs(pos[i]) > 1e-12:
            sig_idx = i - 1
            if 0 <= sig_idx < len(vol):
                key = str(vol[sig_idx])
                buckets[key] = buckets.get(key, 0) + 1
    return dict(sorted(buckets.items()))


__all__ = [
    "apply_walk_forward_entry_regime_blocks",
    "trades_by_entry_vol_regime",
]
