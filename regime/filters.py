"""
Session and volatility filters for gating signals or segmenting analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_allowed_mask(
    session_tags: pd.Series | np.ndarray,
    vol_states: pd.Series | np.ndarray,
    *,
    allowed_sessions: tuple[str, ...] | None,
    allowed_volatility_states: tuple[str, ...] | None,
    require_both: bool = True,
) -> np.ndarray:
    """
    Boolean mask (per bar): whether trading / analytics is allowed under filter rules.

    Empty ``allowed_sessions`` → no session restriction.
    Empty ``allowed_volatility_states`` → no vol restriction.

    If both non-empty and ``require_both``, bar must match one value in each set.
    If only one set non-empty, only that dimension is applied.
    """
    sess = np.asarray(session_tags.astype(str))
    vols = np.asarray(vol_states.astype(str))
    n = len(sess)
    if len(vols) != n:
        raise ValueError("session_tags and vol_states length mismatch")

    allow_s = np.ones(n, dtype=bool)
    if allowed_sessions:
        sset = frozenset(allowed_sessions)
        allow_s = np.array([s in sset for s in sess], dtype=bool)

    allow_v = np.ones(n, dtype=bool)
    if allowed_volatility_states:
        vset = frozenset(allowed_volatility_states)
        allow_v = np.array([v in vset for v in vols], dtype=bool)

    if allowed_sessions and allowed_volatility_states:
        return allow_s & allow_v if require_both else allow_s | allow_v
    if allowed_sessions:
        return allow_s
    if allowed_volatility_states:
        return allow_v
    return np.ones(n, dtype=bool)


def apply_signal_gate(signals: np.ndarray, bar_mask: np.ndarray) -> np.ndarray:
    """Zero out signals where ``bar_mask`` is False (same length)."""
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    m = np.asarray(bar_mask, dtype=bool).reshape(-1)
    if sig.shape[0] != m.shape[0]:
        raise ValueError("signals and bar_mask must have same length")
    return np.where(m, sig, 0.0)
