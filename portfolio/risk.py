"""
Risk controls: drawdown-based deleveraging, exposure caps, turnover / risk per trade.
"""

from __future__ import annotations

import numpy as np


def drawdown_exposure_factor(
    equity: float,
    peak_equity: float,
    *,
    soft_start: float = 0.10,
    hard_stop: float = 0.20,
) -> float:
    """
    Multiplier in ``[0, 1]`` to scale portfolio weights as drawdown worsens.

    - Below ``soft_start`` drawdown: factor 1.
    - Between ``soft_start`` and ``hard_stop``: linear decay to 0.
    - At or above ``hard_stop``: 0.

    Drawdown is ``(peak - equity) / peak`` (positive when underwater).
    """
    if soft_start < 0 or hard_stop <= soft_start:
        raise ValueError("need 0 <= soft_start < hard_stop")
    if peak_equity <= 0:
        return 0.0
    dd = (peak_equity - float(equity)) / peak_equity
    if dd <= soft_start:
        return 1.0
    if dd >= hard_stop:
        return 0.0
    return float(1.0 - (dd - soft_start) / (hard_stop - soft_start))


def apply_exposure_limits(
    weights: np.ndarray,
    *,
    max_gross: float | None = None,
    max_net: float | None = None,
    iterations: int = 5,
) -> np.ndarray:
    """
    Enforce gross (sum abs) and net (sum) exposure caps by iterative rescaling.

    Alternates gross and net projection; converges for typical long-only or
    moderate hedge cases. If ``max_net`` is None, net is unconstrained.
    """
    w = np.asarray(weights, dtype=np.float64).copy().reshape(-1)

    for _ in range(iterations):
        if max_gross is not None and max_gross >= 0:
            g = np.sum(np.abs(w))
            if g > max_gross and g > 1e-15:
                w *= max_gross / g

        if max_net is not None and max_net >= 0:
            net = np.sum(w)
            an = abs(net)
            if an > max_net > 1e-15:
                w *= max_net / an

    return w


def apply_max_risk_per_trade(
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    *,
    max_delta_gross: float,
) -> np.ndarray:
    """
    Cap **one-step** change in gross exposure: ``sum |w_new - w_old| <= max_delta_gross``.

    Moves linearly toward ``target_weights`` when the budget is exceeded.
    """
    if max_delta_gross < 0:
        raise ValueError("max_delta_gross must be non-negative")
    w0 = np.asarray(prev_weights, dtype=np.float64).reshape(-1)
    w1 = np.asarray(target_weights, dtype=np.float64).reshape(-1)
    if w0.shape != w1.shape:
        raise ValueError("prev and target must match shape")

    delta = w1 - w0
    dg = float(np.sum(np.abs(delta)))
    if dg <= max_delta_gross or dg < 1e-15:
        return w1.copy()
    alpha = max_delta_gross / dg
    return w0 + delta * alpha


def update_peak_equity(peak: float, equity: float) -> float:
    """Running peak for drawdown calculations."""
    return float(max(peak, equity))


def max_weight_delta_gross_from_notional(max_notional: float, equity: float) -> float:
    """
    Convert a dollar **turnover** cap into ``max_delta_gross`` for :func:`apply_max_risk_per_trade`.

    With weights defined as fraction of equity deployed per name, ``sum |Δw| ≈ notional / equity``.
    """
    if equity <= 0:
        return 0.0
    if max_notional < 0:
        raise ValueError("max_notional must be non-negative")
    return float(max_notional / equity)
