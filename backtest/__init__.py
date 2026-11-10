"""Backtest: causal signals, next-open execution, costs, and performance metrics."""

from backtest.config import BacktestConfig, ExecutionCosts, PositionSizing
from backtest.engine import BacktestEngine, BacktestResult
from backtest.execution import (
    TradeLeg,
    buy_fill_price,
    rebalance_to_target,
    sell_fill_price,
    target_shares,
)
from backtest.metrics import (
    BacktestMetrics,
    max_drawdown,
    period_returns,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    win_rate,
)
from backtest.segmentation import compare_backtest_segments, summarize_backtest_segment
from backtest.signals import SignalGenerator, constant_signal, validate_signals

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "compare_backtest_segments",
    "ExecutionCosts",
    "PositionSizing",
    "SignalGenerator",
    "TradeLeg",
    "buy_fill_price",
    "constant_signal",
    "max_drawdown",
    "period_returns",
    "profit_factor",
    "rebalance_to_target",
    "sell_fill_price",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize",
    "summarize_backtest_segment",
    "target_shares",
    "validate_signals",
    "win_rate",
]
