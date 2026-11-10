"""Aggregate paper session stats for comparison reports (Phase 16.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.config import ExecutionAssumptions
from backtest.metrics import max_drawdown
from live.paper_broker import PaperFill


def summarize_paper_fills(
    fills: list[PaperFill],
    *,
    initial_equity: float,
    final_equity: float,
    assumptions: ExecutionAssumptions,
    date_start: pd.Timestamp | None,
    date_end: pd.Timestamp | None,
    equity_curve: list[float] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary aligned with backtest ``metrics.json`` fields."""
    closes = [f for f in fills if f.realized_pnl is not None]
    pnls = [float(f.realized_pnl) for f in closes if f.realized_pnl is not None]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    win_rate = wins / n if n else 0.0
    total_pnl = float(sum(pnls)) if pnls else float(final_equity - initial_equity)
    avg_trade = float(np.mean(pnls)) if pnls else 0.0
    dd = 0.0
    if equity_curve and len(equity_curve) > 1:
        dd = max_drawdown(np.asarray(equity_curve, dtype=np.float64))
    return {
        "assumptions": assumptions.to_dict(),
        "date_start": str(date_start) if date_start is not None else None,
        "date_end": str(date_end) if date_end is not None else None,
        "n_trades": n,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_trade_pnl": avg_trade,
        "max_drawdown": dd,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
    }
