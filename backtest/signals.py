"""
Signal generation vs execution: contracts and validation.

**Causal contract:** ``signals[t]`` is known only after bar ``t`` closes. The engine
executes toward that target at bar ``t+1`` **open** (no future OHLC in the decision).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SignalGenerator(Protocol):
    """Produces a per-bar target direction; keep separate from fill simulation."""

    def generate(self, context: Any) -> np.ndarray:
        """Return array of length ``n_bars`` with values in ``{-1, 0, 1}`` (or float targets)."""
        ...


def validate_signals(signals: np.ndarray, n_bars: int) -> np.ndarray:
    """
    Check length and finite values. Signals are **targets** in {-1, 0, 1} for discrete strategies.

    Fractional signals are allowed for advanced sizers (interpreted as scaled direction).
    """
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    if sig.shape[0] != n_bars:
        raise ValueError(f"signals length {sig.shape[0]} != n_bars {n_bars}")
    if not np.all(np.isfinite(sig)):
        raise ValueError("signals must be finite")
    if np.max(np.abs(sig)) > 1.000001:
        raise ValueError("signals must be in [-1, 1] for this engine")
    return sig


def constant_signal(value: float, n_bars: int) -> np.ndarray:
    """Helper for tests: flat signal series."""
    out = np.full(n_bars, value, dtype=np.float64)
    return validate_signals(out, n_bars)
