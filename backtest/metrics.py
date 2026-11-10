"""
Performance metrics from an equity curve and realized trade legs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest.execution import TradeLeg


def period_returns(equity: np.ndarray) -> np.ndarray:
    """Simple returns r_t = (E_t - E_{t-1}) / E_{t-1}. Length len(equity)-1."""
    e = np.asarray(equity, dtype=np.float64)
    if len(e) < 2:
        return np.array([], dtype=np.float64)
    prev = e[:-1]
    out = (e[1:] - prev) / np.where(prev != 0, prev, np.nan)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def sharpe_ratio(
    returns: np.ndarray,
    *,
    bars_per_year: int = 252,
    risk_free_annual: float = 0.0,
) -> float:
    """Annualized Sharpe using sample std (ddof=0)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    rf_bar = risk_free_annual / float(bars_per_year)
    x = r - rf_bar
    sd = float(np.std(x))
    if sd < 1e-15:
        return 0.0
    return float(np.sqrt(bars_per_year) * np.mean(x) / sd)


def sortino_ratio(
    returns: np.ndarray,
    *,
    bars_per_year: int = 252,
    risk_free_annual: float = 0.0,
) -> float:
    """Annualized Sortino: downside deviation of returns below target (rf per bar)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    rf_bar = risk_free_annual / float(bars_per_year)
    x = r - rf_bar
    downside = np.minimum(x, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd < 1e-15:
        return 0.0
    return float(np.sqrt(bars_per_year) * np.mean(x) / dd)


def max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline as a **negative** fraction (e.g. -0.2 for 20% drawdown).
    """
    e = np.asarray(equity, dtype=np.float64)
    if e.size < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / np.where(peak > 0, peak, np.nan)
    return float(np.nanmin(dd))


def win_rate(trade_pnls: np.ndarray) -> float:
    """Share of trades with strictly positive PnL (zero if no trades)."""
    p = np.asarray(trade_pnls, dtype=np.float64)
    if p.size == 0:
        return 0.0
    wins = np.sum(p > 0)
    return float(wins / p.size)


def profit_factor(trade_pnls: np.ndarray) -> float:
    """Sum of wins / abs(sum of losses); inf if no losses; nan if no trades."""
    p = np.asarray(trade_pnls, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    wins = np.sum(p[p > 0])
    losses = np.sum(p[p < 0])
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def trade_pnls_from_legs(legs: list[TradeLeg]) -> np.ndarray:
    return np.array([lg.pnl for lg in legs], dtype=np.float64)


@dataclass
class BacktestMetrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    """Fraction of trade legs with strictly positive ``TradeLeg.pnl`` (fill-price PnL)."""
    profit_factor: float
    """Win/loss ratio from ``TradeLeg.pnl`` (includes spread/slippage via fills; excludes commissions)."""
    total_return: float
    n_trades: int
    """Count of ``TradeLeg`` closed-lot records with realized ``pnl`` (not open-only fills)."""


def summarize(
    equity: np.ndarray,
    trade_legs: list[TradeLeg],
    *,
    bars_per_year: int = 252,
    risk_free_annual: float = 0.0,
) -> BacktestMetrics:
    """Bundle common metrics for reporting."""
    e = np.asarray(equity, dtype=np.float64)
    r = period_returns(e)
    pnls = trade_pnls_from_legs(trade_legs)
    tr = (e[-1] / e[0] - 1.0) if e.size > 0 and e[0] > 0 else 0.0
    pf = profit_factor(pnls)
    if np.isinf(pf) or np.isnan(pf):
        pf_out = pf
    else:
        pf_out = float(pf)
    return BacktestMetrics(
        sharpe=sharpe_ratio(r, bars_per_year=bars_per_year, risk_free_annual=risk_free_annual),
        sortino=sortino_ratio(r, bars_per_year=bars_per_year, risk_free_annual=risk_free_annual),
        max_drawdown=max_drawdown(e),
        win_rate=win_rate(pnls),
        profit_factor=pf_out,
        total_return=float(tr),
        # n_trades counts closed-lot TradeLeg records (see TradeLeg docstring).
        n_trades=len(trade_legs),
    )
