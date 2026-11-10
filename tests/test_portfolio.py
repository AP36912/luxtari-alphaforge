"""Portfolio sizing, Kelly, risk limits, and multi-asset types."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.constraints import RiskBudget, apply_risk_budget
from portfolio.kelly import safe_kelly_fraction
from portfolio.pipeline import PortfolioPolicy, build_target_weights
from portfolio.risk import (
    apply_exposure_limits,
    apply_max_risk_per_trade,
    drawdown_exposure_factor,
    max_weight_delta_gross_from_notional,
    update_peak_equity,
)
from portfolio.sizing import portfolio_vol_diagonal, weights_fixed_fraction, weights_volatility_target
from portfolio.types import AssetUniverse, PortfolioWeights, validate_square_cov


def test_weights_fixed_fraction() -> None:
    s = np.array([1.0, -0.5, 0.0])
    w = weights_fixed_fraction(s, 0.2)
    np.testing.assert_allclose(w, [0.2, -0.1, 0.0])


def test_weights_volatility_target_scales_to_target_vol() -> None:
    signals = np.array([1.0, 1.0])
    vol = np.array([0.20, 0.20])
    w = weights_volatility_target(signals, vol, target_vol_annual=0.10, max_gross=2.0)
    est = portfolio_vol_diagonal(w, vol)
    assert est == pytest.approx(0.10, rel=0, abs=0.02)


def test_apply_exposure_limits_gross_and_net() -> None:
    w = np.array([0.5, 0.5, -0.4])
    out = apply_exposure_limits(w, max_gross=1.0, max_net=0.2, iterations=10)
    assert np.sum(np.abs(out)) <= 1.0 + 1e-9
    assert abs(np.sum(out)) <= 0.2 + 1e-6


def test_apply_max_risk_per_trade_caps_delta() -> None:
    w0 = np.zeros(3)
    w1 = np.array([0.5, 0.3, -0.2])
    out = apply_max_risk_per_trade(w0, w1, max_delta_gross=0.5)
    dg = np.sum(np.abs(out - w0))
    assert dg <= 0.5 + 1e-9
    # direction preserved when alpha < 1
    np.testing.assert_allclose(out / np.linalg.norm(out), w1 / np.linalg.norm(w1), atol=1e-6)


def test_drawdown_exposure_factor() -> None:
    assert drawdown_exposure_factor(100.0, 100.0, soft_start=0.1, hard_stop=0.2) == 1.0
    assert drawdown_exposure_factor(80.0, 100.0, soft_start=0.1, hard_stop=0.2) == 0.0  # 20% dd
    mid = drawdown_exposure_factor(88.0, 100.0, soft_start=0.1, hard_stop=0.2)  # 12% dd
    assert 0.0 < mid < 1.0


def test_update_peak_equity() -> None:
    assert update_peak_equity(100.0, 90.0) == 100.0
    assert update_peak_equity(100.0, 110.0) == 110.0


def test_max_weight_delta_gross_from_notional() -> None:
    assert max_weight_delta_gross_from_notional(5_000.0, 100_000.0) == pytest.approx(0.05)


def test_safe_kelly_fraction_quarter_kelly_capped() -> None:
    # p=0.6, b=1 -> raw = (0.6-0.4)/1 = 0.2, quarter = 0.05
    fk = safe_kelly_fraction(0.6, 1.0, 1.0, kelly_fraction=0.25, cap=1.0)
    assert fk == pytest.approx(0.05)
    fk_capped = safe_kelly_fraction(0.9, 2.0, 0.5, kelly_fraction=1.0, cap=0.1)
    assert fk_capped <= 0.1 + 1e-9


def test_safe_kelly_non_positive_returns_floor() -> None:
    assert safe_kelly_fraction(0.4, 1.0, 1.0, kelly_fraction=0.25, cap=0.5) == 0.0


def test_risk_budget_clips_names_and_gross() -> None:
    budget = RiskBudget(
        max_gross_exposure=0.5,
        max_single_name_weight=0.2,
        max_net_exposure=0.3,
    )
    w = np.array([0.4, 0.4, 0.4])
    out = apply_risk_budget(w, budget)
    assert np.max(np.abs(out)) <= 0.2 + 1e-9
    assert np.sum(np.abs(out)) <= 0.5 + 1e-8


def test_build_target_weights_drawdown_zeroes() -> None:
    pol = PortfolioPolicy(
        sizing_mode="fixed_fraction",
        fixed_fraction=0.5,
        max_gross=1.0,
        drawdown_soft_start=0.05,
        drawdown_hard_stop=0.15,
    )
    w = build_target_weights(
        np.array([1.0]),
        policy=pol,
        equity=80.0,
        peak_equity=100.0,
    )
    assert np.allclose(w, 0.0)


def test_build_target_weights_with_turnover_cap() -> None:
    pol = PortfolioPolicy(
        fixed_fraction=1.0,
        max_gross=1.0,
        max_delta_gross=0.2,
    )
    w = build_target_weights(
        np.array([1.0, 0.0]),
        policy=pol,
        equity=100.0,
        peak_equity=100.0,
        prev_weights=np.zeros(2),
    )
    assert np.sum(np.abs(w)) <= 0.2 + 1e-9


def test_asset_universe_and_portfolio_weights() -> None:
    u = AssetUniverse(ids=("A", "B"))
    pw = PortfolioWeights(universe=u, values=np.array([0.1, -0.2]))
    assert pw.gross_exposure == pytest.approx(0.3)
    assert pw.as_dict() == {"A": 0.1, "B": -0.2}
    with pytest.raises(KeyError):
        _ = u.index("Z")


def test_validate_square_cov() -> None:
    c = np.eye(3)
    validate_square_cov(c, 3)
    with pytest.raises(ValueError):
        validate_square_cov(np.eye(2), 3)


def test_portfolio_policy_rejects_invalid_drawdown_bounds() -> None:
    with pytest.raises(ValueError, match="drawdown"):
        PortfolioPolicy(drawdown_soft_start=0.2, drawdown_hard_stop=0.15)
