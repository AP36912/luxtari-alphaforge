"""
Portfolio constraints and risk budgets (gross/net, per-name caps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio.risk import apply_exposure_limits


@dataclass(frozen=True)
class RiskBudget:
    """
    Static limits applied to a weight vector (any number of assets).

    ``max_single_name_weight`` caps ``|w_i|`` before gross/net rescaling.
    """

    max_gross_exposure: float
    max_single_name_weight: float
    max_net_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.max_single_name_weight <= 0:
            raise ValueError("max_single_name_weight must be positive")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be non-negative if set")


def apply_risk_budget(weights: np.ndarray, budget: RiskBudget) -> np.ndarray:
    """
    Clip per-name weights, then enforce gross (and optional net) exposure.
    """
    w = np.asarray(weights, dtype=np.float64).copy().reshape(-1)
    lim = budget.max_single_name_weight
    w = np.clip(w, -lim, lim)
    return apply_exposure_limits(
        w,
        max_gross=budget.max_gross_exposure,
        max_net=budget.max_net_exposure,
    )
