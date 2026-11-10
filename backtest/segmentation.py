"""
Slice backtest metrics by bar-level masks (sessions, volatility regimes, configured gates).

Equity Sharpe/Sortino use **paired** bar returns where both endpoints are inside the mask;
trade statistics use legs whose ``bar`` index is masked in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtest.execution import TradeLeg
from backtest.metrics import (
    max_drawdown,
    period_returns,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    trade_pnls_from_legs,
    win_rate,
)


def summarize_backtest_segment(
    equity: np.ndarray,
    trade_legs: list[TradeLeg],
    bar_mask: np.ndarray,
    *,
    bars_per_year: int,
    risk_free_annual: float = 0.0,
) -> dict[str, Any]:
    """
    Metrics restricted to bars in ``bar_mask`` (length == len(equity)).

    - Returns: consecutive pairs inside mask.
    - Trades: legs with ``leg.bar`` True in mask.
    - Total return: compounded masked simple returns (subset sequence).
    - Drawdown: on equity subseries ``equity[mask]`` (indices where mask True).
    """
    e = np.asarray(equity, dtype=np.float64)
    m = np.asarray(bar_mask, dtype=bool).reshape(-1)
    if e.size != m.size:
        raise ValueError("equity and bar_mask length mismatch")

    r = period_returns(e)
    if r.size >= 1 and m.size >= 2:
        pair = m[:-1] & m[1:]
        mr = r[pair]
    else:
        mr = np.array([], dtype=np.float64)

    sharpe = sharpe_ratio(mr, bars_per_year=bars_per_year, risk_free_annual=risk_free_annual)
    sortino = sortino_ratio(mr, bars_per_year=bars_per_year, risk_free_annual=risk_free_annual)
    if mr.size:
        total_ret = float(np.prod(1.0 + mr) - 1.0)
    else:
        total_ret = 0.0

    e_sub = e[m]
    dd = max_drawdown(e_sub) if e_sub.size > 1 else 0.0

    sub_legs = [lg for lg in trade_legs if 0 <= int(lg.bar) < m.size and m[int(lg.bar)]]
    pnls = trade_pnls_from_legs(sub_legs)
    wr = win_rate(pnls)
    pf = profit_factor(pnls)
    if np.isinf(pf) or np.isnan(pf):
        pf_out = float(pf) if np.isinf(pf) else 0.0
    else:
        pf_out = float(pf)

    return {
        "n_bars_in_mask": int(m.sum()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "win_rate": float(wr),
        "profit_factor": float(pf_out),
        "total_return": float(total_ret),
        "n_trades": len(sub_legs),
    }


def compare_backtest_segments(
    equity: np.ndarray,
    trade_legs: list[TradeLeg],
    segment_masks: dict[str, np.ndarray],
    *,
    bars_per_year: int,
    risk_free_annual: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Run :func:`summarize_backtest_segment` for each named mask."""
    out: dict[str, dict[str, Any]] = {}
    for name, mask in segment_masks.items():
        out[name] = summarize_backtest_segment(
            equity,
            trade_legs,
            mask,
            bars_per_year=bars_per_year,
            risk_free_annual=risk_free_annual,
        )
    return out
