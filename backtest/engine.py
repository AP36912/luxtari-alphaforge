"""
Backtest engine: causal signal → execution at configurable bar price, costs, equity path.

**No future knowledge:** default ``fill_basis=next_bar_open``: ``signals[t]`` is acted on at
``open[t+1]``; sizing uses equity marked at ``close[t]``. With ``bar_close``, ``signals[t]``
executes at ``close[t]`` using the same fractional cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.execution import TradeLeg, rebalance_to_target, target_shares
from backtest.signals import validate_signals


@dataclass
class BacktestResult:
    """Outputs from :meth:`BacktestEngine.run`."""

    equity: np.ndarray
    cash: np.ndarray
    position: np.ndarray
    open: np.ndarray
    close: np.ndarray
    signals: np.ndarray
    realized_pnl: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    unrealized_pnl: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    trade_legs: list[TradeLeg] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)


class BacktestEngine:
    """
    Vectorized bar simulation with fills driven by :class:`~backtest.config.ExecutionAssumptions`.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config if config is not None else BacktestConfig()

    def run(
        self,
        ohlc: pd.DataFrame | Mapping[str, Any],
        signals: np.ndarray,
    ) -> BacktestResult:
        if isinstance(ohlc, pd.DataFrame):
            open_ = np.asarray(ohlc["open"], dtype=np.float64)
            high = np.asarray(ohlc.get("high", ohlc["close"]), dtype=np.float64)
            low = np.asarray(ohlc.get("low", ohlc["close"]), dtype=np.float64)
            close = np.asarray(ohlc["close"], dtype=np.float64)
        else:
            open_ = np.asarray(ohlc["open"], dtype=np.float64)
            high = np.asarray(ohlc.get("high", ohlc["close"]), dtype=np.float64)
            low = np.asarray(ohlc.get("low", ohlc["close"]), dtype=np.float64)
            close = np.asarray(ohlc["close"], dtype=np.float64)

        n = len(close)
        if len(open_) != n:
            raise ValueError("open and close must have same length")
        sig = validate_signals(signals, n)

        cfg = self.config
        cash = float(cfg.initial_capital)
        pos = 0.0
        avg = 0.0

        equity = np.zeros(n, dtype=np.float64)
        cash_hist = np.zeros(n, dtype=np.float64)
        pos_hist = np.zeros(n, dtype=np.float64)
        realized_hist = np.zeros(n, dtype=np.float64)
        unrealized_hist = np.zeros(n, dtype=np.float64)
        legs: list[TradeLeg] = []

        basis = cfg.assumptions.fill_basis
        costs = cfg.assumptions.costs

        def apply_intrabar_stop(i: int) -> None:
            nonlocal pos, avg, cash
            if abs(pos) < 1e-12:
                return
            is_long = pos > 0
            stop = (
                avg * (1.0 - cfg.stop_loss_pct) if is_long and cfg.stop_loss_pct is not None
                else avg * (1.0 + cfg.stop_loss_pct) if (not is_long and cfg.stop_loss_pct is not None)
                else None
            )
            target = (
                avg * (1.0 + cfg.take_profit_pct) if is_long and cfg.take_profit_pct is not None
                else avg * (1.0 - cfg.take_profit_pct) if (not is_long and cfg.take_profit_pct is not None)
                else None
            )
            ref: float | None = None
            if is_long and stop is not None and low[i] <= stop:
                ref = min(stop, low[i]) if cfg.assumptions.stop_target_fill_policy == "bar_extreme" else stop
            elif is_long and target is not None and high[i] >= target:
                ref = max(target, high[i]) if cfg.assumptions.stop_target_fill_policy == "bar_extreme" else target
            elif not is_long and stop is not None and high[i] >= stop:
                ref = max(stop, high[i]) if cfg.assumptions.stop_target_fill_policy == "bar_extreme" else stop
            elif not is_long and target is not None and low[i] <= target:
                ref = min(target, low[i]) if cfg.assumptions.stop_target_fill_policy == "bar_extreme" else target
            if ref is not None:
                pos, avg, cash, stopped = rebalance_to_target(
                    bar_index=i, open_px=float(ref), position=pos, avg_entry=avg,
                    cash=cash, target_shares_=0.0, costs=costs,
                )
                legs.extend(stopped)

        if basis == "next_bar_open":
            for i in range(n):
                if i > 0:
                    equity_at_decision = cash + pos * close[i - 1]
                    tgt = target_shares(
                        sig[i - 1],
                        equity_at_decision,
                        float(open_[i]),
                        cfg.sizing,
                    )
                    pos, avg, cash, new_legs = rebalance_to_target(
                        bar_index=i,
                        open_px=float(open_[i]),
                        position=pos,
                        avg_entry=avg,
                        cash=cash,
                        target_shares_=tgt,
                        costs=costs,
                    )
                    legs.extend(new_legs)

                apply_intrabar_stop(i)
                equity[i] = cash + pos * close[i]
                cash_hist[i] = cash
                pos_hist[i] = pos
                realized_hist[i] = sum(leg.pnl for leg in legs)
                unrealized_hist[i] = pos * (close[i] - avg)
        else:
            # bar_close: signal[i] executes at close[i] after marking at close[i]
            for i in range(n):
                apply_intrabar_stop(i)
                E = cash + pos * close[i]
                tgt = target_shares(sig[i], E, float(close[i]), cfg.sizing)
                pos, avg, cash, new_legs = rebalance_to_target(
                    bar_index=i,
                    open_px=float(close[i]),
                    position=pos,
                    avg_entry=avg,
                    cash=cash,
                    target_shares_=tgt,
                    costs=costs,
                )
                legs.extend(new_legs)
                equity[i] = cash + pos * close[i]
                cash_hist[i] = cash
                pos_hist[i] = pos
                realized_hist[i] = sum(leg.pnl for leg in legs)
                unrealized_hist[i] = pos * (close[i] - avg)

        return BacktestResult(
            equity=equity,
            cash=cash_hist,
            position=pos_hist,
            realized_pnl=realized_hist,
            unrealized_pnl=unrealized_hist,
            open=open_,
            close=close,
            signals=sig,
            trade_legs=legs,
            config=cfg,
        )
