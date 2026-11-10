"""Mid-path parity tests for BacktestEngine and PaperBroker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.config import BacktestConfig, ExecutionAssumptions, ExecutionCosts, PositionSizing
from backtest.engine import BacktestEngine
from live.paper_broker import PaperBroker, paper_broker_config_from_backtest
from reporting.backtest_paper_compare import assumptions_match


def _ohlcv(n: int) -> pd.DataFrame:
    close = np.array([100, 102, 99, 103, 101, 98, 104, 100, 105, 99], dtype=float)[:n]
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_, "high": np.maximum(open_, close) + 0.25,
        "low": np.minimum(open_, close) - 0.25, "close": close,
        "volume": 1000.0,
    })


def _run_pair(targets: list[int], assumptions: ExecutionAssumptions, *, stop_loss_pct=None):
    df = _ohlcv(len(targets))
    cfg = BacktestConfig(
        initial_capital=100_000.0, assumptions=assumptions,
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
        stop_loss_pct=stop_loss_pct,
    )
    bt = BacktestEngine(cfg).run(df, np.asarray(targets, dtype=float))
    paper = PaperBroker(paper_broker_config_from_backtest(cfg))
    cash, equity, pos, realized, unrealized = [], [], [], [], []
    for i, row in df.iterrows():
        paper.step(row.timestamp, row.open, row.high, row.low, row.close, targets[i])
        snap = paper.portfolio_snapshot(float(row.close))
        cash.append(snap["cash"]); equity.append(snap["equity"])
        pos.append(snap["net_exposure_shares"]); realized.append(snap["realized_pnl"])
        unrealized.append(snap["unrealized_pnl"])
    return bt, paper, df, {"cash": np.array(cash), "equity": np.array(equity), "pos": np.array(pos),
                            "realized": np.array(realized), "unrealized": np.array(unrealized)}


@pytest.mark.parametrize("targets", [
    [0, 1, 1, 0, 0, 0], [0, -1, -1, 0, 0, 0],
    [0, 1, -1, 0, 0, 0], [0, -1, 1, 0, 0, 0],
])
@pytest.mark.parametrize("basis", ["next_bar_open", "bar_close"])
def test_long_short_flat_reversal_midpath_parity(targets, basis):
    bt, _paper, _df, ph = _run_pair(targets, ExecutionAssumptions(fill_basis=basis))
    np.testing.assert_allclose(bt.cash, ph["cash"])
    np.testing.assert_allclose(bt.equity, ph["equity"])
    np.testing.assert_allclose(bt.position, ph["pos"])
    np.testing.assert_allclose(bt.realized_pnl, ph["realized"])
    np.testing.assert_allclose(bt.unrealized_pnl, ph["unrealized"])


@pytest.mark.parametrize("costs", [
    ExecutionCosts(commission_rate=0.0005),
    ExecutionCosts(half_spread=0.0002),
    ExecutionCosts(slippage=0.0003),
    ExecutionCosts(half_spread=0.0002, slippage=0.0003, commission_rate=0.0005),
])
def test_isolated_and_combined_cost_parity(costs):
    bt, _paper, _df, ph = _run_pair([0, 1, 1, -1, -1, 0], ExecutionAssumptions(costs=costs))
    np.testing.assert_allclose(bt.cash, ph["cash"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(bt.equity, ph["equity"], rtol=0, atol=1e-9)


def test_stop_loss_shared_policy_parity():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
        "open": [100., 100., 95.], "high": [100., 101., 96.],
        "low": [100., 94., 94.], "close": [100., 95., 95.], "volume": 1.,
    })
    cfg = BacktestConfig(assumptions=ExecutionAssumptions(fill_basis="bar_close", stop_target_fill_policy="stop_level"),
                         sizing=PositionSizing(fixed_shares=1.0), stop_loss_pct=0.05)
    bt = BacktestEngine(cfg).run(df, np.array([1., 0., 0.]))
    paper = PaperBroker(paper_broker_config_from_backtest(cfg))
    paths = []
    for i, r in df.iterrows():
        paper.step(r.timestamp, r.open, r.high, r.low, r.close, int([1, 0, 0][i]))
        paths.append(paper.equity(r.close))
    np.testing.assert_allclose(bt.equity, paths)
    assert any(f.reason == "stop_loss" for f in paper.state.fills)


def test_deterministic_replay():
    a = _run_pair([0, 1, -1, 1, 0, 0], ExecutionAssumptions(costs=ExecutionCosts(slippage=.001)))[0]
    b = _run_pair([0, 1, -1, 1, 0, 0], ExecutionAssumptions(costs=ExecutionCosts(slippage=.001)))[0]
    np.testing.assert_array_equal(a.equity, b.equity)


def test_assumptions_mismatch_is_detected():
    ok, diffs = assumptions_match(ExecutionAssumptions(fill_basis="next_bar_open").to_dict(),
                                  ExecutionAssumptions(fill_basis="bar_close").to_dict())
    assert not ok and diffs


def test_invalid_assumptions_rejected():
    with pytest.raises(ValueError): ExecutionAssumptions(fill_basis="same_bar_open")  # type: ignore[arg-type]
    with pytest.raises(ValueError): ExecutionCosts(half_spread=-0.1)


def test_fixed_share_sizing_greater_than_one_midpath_parity():
    df = _ohlcv(6)
    targets = [0, 1, 1, -1, -1, 0]
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        assumptions=ExecutionAssumptions(fill_basis="next_bar_open"),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=7.0),
    )
    bt = BacktestEngine(cfg).run(df, np.asarray(targets, dtype=float))
    paper = PaperBroker(paper_broker_config_from_backtest(cfg))
    cash, equity, position = [], [], []
    for i, row in df.iterrows():
        paper.step(row.timestamp, row.open, row.high, row.low, row.close, targets[i])
        snap = paper.portfolio_snapshot(float(row.close))
        cash.append(snap["cash"]); equity.append(snap["equity"]); position.append(snap["net_exposure_shares"])
    np.testing.assert_allclose(bt.cash, cash)
    np.testing.assert_allclose(bt.equity, equity)
    np.testing.assert_allclose(bt.position, position)


def test_fraction_of_equity_paper_parity_fails_explicitly():
    cfg = BacktestConfig(sizing=PositionSizing(mode="fraction_of_equity", fraction_of_equity=0.5))
    with pytest.raises(ValueError, match="does not support fraction_of_equity"):
        paper_broker_config_from_backtest(cfg)
