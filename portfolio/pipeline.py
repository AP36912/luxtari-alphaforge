"""
Compose sizing + Kelly scaling + risk budget + drawdown + turnover caps.

Designed for **per-bar** or **per-rebalance** calls in multi-asset loops: same API
for ``n_assets == 1`` and larger universes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from portfolio.constraints import RiskBudget, apply_risk_budget
from portfolio.kelly import safe_kelly_fraction
from portfolio.risk import (
    apply_exposure_limits,
    apply_max_risk_per_trade,
    drawdown_exposure_factor,
)
from portfolio.sizing import scale_weights_by_kelly, weights_fixed_fraction, weights_volatility_target

SizingMode = Literal["fixed_fraction", "vol_target"]


@dataclass(frozen=True)
class PortfolioPolicy:
    """Knobs for :func:`build_target_weights`."""

    sizing_mode: SizingMode = "fixed_fraction"
    fixed_fraction: float = 0.2
    target_vol_annual: float = 0.15
    max_gross: float = 1.0
    max_net: float | None = None
    max_delta_gross: float | None = None
    drawdown_soft_start: float = 0.10
    drawdown_hard_stop: float = 0.20
    kelly_multiplier: float = 1.0
    risk_budget: RiskBudget | None = None

    def __post_init__(self) -> None:
        if self.kelly_multiplier < 0:
            raise ValueError("kelly_multiplier must be non-negative")
        if self.max_delta_gross is not None and self.max_delta_gross < 0:
            raise ValueError("max_delta_gross must be non-negative if set")
        if self.drawdown_soft_start < 0 or self.drawdown_hard_stop <= self.drawdown_soft_start:
            raise ValueError("need 0 <= drawdown_soft_start < drawdown_hard_stop")


def build_target_weights(
    signals: np.ndarray,
    *,
    policy: PortfolioPolicy,
    equity: float,
    peak_equity: float,
    vol_annual: np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full pipeline: base sizing → Kelly scale → risk budget → drawdown factor → turnover cap.

    ``vol_annual`` required when ``policy.sizing_mode == \"vol_target\"``.
    ``prev_weights`` required when ``policy.max_delta_gross`` is set.
    """
    s = np.asarray(signals, dtype=np.float64).reshape(-1)

    if policy.sizing_mode == "fixed_fraction":
        w = weights_fixed_fraction(s, policy.fixed_fraction)
    else:
        if vol_annual is None:
            raise ValueError("vol_annual required for vol_target sizing")
        w = weights_volatility_target(
            s,
            np.asarray(vol_annual, dtype=np.float64).reshape(-1),
            policy.target_vol_annual,
            max_gross=policy.max_gross,
        )

    if policy.kelly_multiplier != 1.0:
        w = scale_weights_by_kelly(w, policy.kelly_multiplier)

    if policy.risk_budget is not None:
        w = apply_risk_budget(w, policy.risk_budget)
    else:
        w = apply_exposure_limits(
            w,
            max_gross=policy.max_gross,
            max_net=policy.max_net,
        )

    dd_f = drawdown_exposure_factor(
        equity,
        peak_equity,
        soft_start=policy.drawdown_soft_start,
        hard_stop=policy.drawdown_hard_stop,
    )
    w = w * dd_f

    if policy.max_delta_gross is not None:
        if prev_weights is None:
            raise ValueError("prev_weights required when max_delta_gross is set")
        w = apply_max_risk_per_trade(
            np.asarray(prev_weights, dtype=np.float64).reshape(-1),
            w,
            max_delta_gross=policy.max_delta_gross,
        )

    return w


def kelly_multiplier_from_history(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    **kwargs: float,
) -> float:
    """Thin wrapper returning a safe Kelly fraction for use as ``policy.kelly_multiplier``."""
    return safe_kelly_fraction(win_rate, avg_win, avg_loss, **kwargs)
