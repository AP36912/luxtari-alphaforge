"""Backtest PnL, costs, and causal execution (next-bar open fills)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.config import BacktestConfig, ExecutionAssumptions, ExecutionCosts, PositionSizing
from backtest.engine import BacktestEngine
from backtest.execution import (
    buy_fill_price,
    rebalance_to_target,
    sell_fill_price,
)
from backtest.metrics import max_drawdown, period_returns, summarize
from backtest.signals import validate_signals


def _ohlc(open_: list[float], high: list[float], low: list[float], close: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_long_round_trip_pnl_zero_costs() -> None:
    """Buy at open[1], sell at open[2]; PnL = sell_fill - buy_fill on one share."""
    df = _ohlc([100.0, 100.0, 102.0], [100, 100, 102], [100, 100, 102], [100.0, 100.0, 102.0])
    sig = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        assumptions=ExecutionAssumptions(costs=ExecutionCosts()),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
    )
    eng = BacktestEngine(cfg)
    res = eng.run(df, sig)
    buy_px = buy_fill_price(100.0, cfg.costs)
    sell_px = sell_fill_price(102.0, cfg.costs)
    expected_equity = 100_000.0 - buy_px + sell_px
    assert res.equity[-1] == pytest.approx(expected_equity, rel=0, abs=1e-6)
    assert res.position[-1] == pytest.approx(0.0)


def test_short_round_trip_pnl_zero_costs() -> None:
    """Short at higher open, cover at lower open."""
    df = _ohlc([100.0, 110.0, 100.0], [110, 110, 100], [100, 100, 100], [100.0, 110.0, 100.0])
    sig = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    cfg = BacktestConfig(
        initial_capital=50_000.0,
        assumptions=ExecutionAssumptions(costs=ExecutionCosts()),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
    )
    eng = BacktestEngine(cfg)
    res = eng.run(df, sig)
    short_px = sell_fill_price(110.0, cfg.costs)
    cover_px = buy_fill_price(100.0, cfg.costs)
    # Open short: receive short_px; cover: pay cover_px
    expected = 50_000.0 + short_px - cover_px
    assert res.equity[-1] == pytest.approx(expected, rel=0, abs=1e-6)


def test_commission_reduces_final_equity() -> None:
    df = _ohlc([100.0, 100.0, 100.0], [100, 100, 100], [100, 100, 100], [100.0, 100.0, 100.0])
    sig = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    no_comm = BacktestConfig(
        initial_capital=10_000.0,
        assumptions=ExecutionAssumptions(costs=ExecutionCosts(commission_rate=0.0)),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
    )
    with_comm = BacktestConfig(
        initial_capital=10_000.0,
        assumptions=ExecutionAssumptions(costs=ExecutionCosts(commission_rate=0.01)),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
    )
    e0 = BacktestEngine(no_comm).run(df, sig).equity[-1]
    e1 = BacktestEngine(with_comm).run(df, sig).equity[-1]
    assert e1 < e0
    # One round trip: buy notional ~100, sell ~100 -> commission ~1 + ~1 = 2 on 1% rate
    assert e0 - e1 == pytest.approx(2.0, rel=0, abs=0.05)


def test_spread_and_slippage_widen_buy_tighten_sell() -> None:
    o = 100.0
    costs = ExecutionCosts(half_spread=0.001, slippage=0.001)
    assert buy_fill_price(o, costs) > o
    assert sell_fill_price(o, costs) < o


def test_bar_close_executes_at_close_not_next_open() -> None:
    """With fill_basis=bar_close, first-bar signal trades at close[0] (not open[1])."""
    df = _ohlc([50.0, 100.0, 100.0], [100, 100, 100], [50, 50, 50], [100.0, 100.0, 100.0])
    sig = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        assumptions=ExecutionAssumptions(fill_basis="bar_close", costs=ExecutionCosts()),
        sizing=PositionSizing(mode="fixed_shares", fixed_shares=1.0),
    )
    res = BacktestEngine(cfg).run(df, sig)
    from backtest.execution import buy_fill_price, sell_fill_price

    buy_px = buy_fill_price(100.0, cfg.costs)
    sell_px = sell_fill_price(100.0, cfg.costs)
    expected = 10_000.0 - buy_px + sell_px
    assert res.equity[-1] == pytest.approx(expected, rel=0, abs=1e-6)


def test_execution_assumptions_rejects_unknown_fill_basis() -> None:
    with pytest.raises(ValueError, match="fill_basis"):
        ExecutionAssumptions(fill_basis="midpoint")  # type: ignore[arg-type]


def test_signal_at_last_bar_not_executed_no_future_bar() -> None:
    """Final signal cannot use future open — engine only trades from bar 1..n-1."""
    df = _ohlc([10.0, 10.0], [10, 10], [10, 10], [10.0, 10.0])
    sig = np.array([0.0, 1.0], dtype=np.float64)
    res = BacktestEngine(BacktestConfig(sizing=PositionSizing(fixed_shares=1.0))).run(df, sig)
    assert res.position[-1] == pytest.approx(0.0)
    assert res.equity[-1] == pytest.approx(100_000.0)


def test_fraction_of_equity_sizing_long() -> None:
    df = _ohlc(
        [100.0, 100.0, 100.0, 100.0],
        [100, 100, 100, 100],
        [100, 100, 100, 100],
        [100.0, 100.0, 100.0, 100.0],
    )
    sig = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        assumptions=ExecutionAssumptions(costs=ExecutionCosts()),
        sizing=PositionSizing(mode="fraction_of_equity", fraction_of_equity=0.5),
    )
    res = BacktestEngine(cfg).run(df, sig)
    # Bar 2 open: equity at prior close ~10k → half → 50 shares @ 100; flat on next signal
    assert res.position[1] == pytest.approx(0.0)
    assert res.position[2] == pytest.approx(50.0, rel=0, abs=1e-9)
    assert res.position[3] == pytest.approx(0.0)


def test_validate_signals_length() -> None:
    with pytest.raises(ValueError, match="length"):
        validate_signals(np.array([1.0]), 3)


def test_rebalance_flip_long_to_short_realizes_long_then_opens_short() -> None:
    costs = ExecutionCosts()
    pos, avg, cash, legs = rebalance_to_target(
        bar_index=1,
        open_px=100.0,
        position=1.0,
        avg_entry=90.0,
        cash=10_000.0,
        target_shares_=-1.0,
        costs=costs,
    )
    assert pos == pytest.approx(-1.0)
    # Closing the long realizes PnL; opening short is not a realized leg until cover
    assert len(legs) == 1
    assert legs[0].pnl == pytest.approx(sell_fill_price(100.0, costs) - 90.0)


def test_metrics_equity_drawdown_and_summarize() -> None:
    eq = np.array([100.0, 110.0, 105.0, 130.0], dtype=np.float64)
    r = period_returns(eq)
    assert r[0] == pytest.approx(0.1)
    assert max_drawdown(eq) < 0
    from backtest.execution import TradeLeg

    m = summarize(eq, [TradeLeg(bar=1, pnl=5.0), TradeLeg(bar=2, pnl=-1.0)])
    assert m.n_trades == 2
    assert m.total_return == pytest.approx(0.3)
