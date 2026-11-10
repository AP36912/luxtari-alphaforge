"""
Backtest agent: strategy simulation via ``backtest.engine`` and ``backtest.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.types import BacktestAgentOutput
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.metrics import summarize


@dataclass
class BacktestAgentInput:
    ohlc_holdout: pd.DataFrame
    """Rows aligned with predictions (must include open, high, low, close)."""
    signals: np.ndarray
    """Model predictions or direction targets (same length as ``ohlc_holdout``)."""
    backtest_config: BacktestConfig


def run_backtest_agent(inp: BacktestAgentInput) -> BacktestAgentOutput:
    """Run :class:`BacktestEngine` and summarize risk/return metrics."""
    engine = BacktestEngine(inp.backtest_config)
    result = engine.run(inp.ohlc_holdout, inp.signals)
    m = summarize(
        result.equity,
        result.trade_legs,
        bars_per_year=inp.backtest_config.bars_per_year,
    )
    return BacktestAgentOutput(
        sharpe=m.sharpe,
        sortino=m.sortino,
        max_drawdown=m.max_drawdown,
        win_rate=m.win_rate,
        profit_factor=m.profit_factor,
        total_return=m.total_return,
        n_trades=m.n_trades,
    )
