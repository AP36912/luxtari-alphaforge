"""
Bundle common figures for a :class:`backtest.engine.BacktestResult` into ``outputs/reports/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_backtest_report(
    result: Any,
    *,
    prefix: str = "backtest",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Write equity curve, drawdown, and trade PnL histogram for one backtest run.

    Args:
        result: :class:`backtest.engine.BacktestResult` instance.
        prefix: Filename prefix.
        output_dir: If set, absolute or relative paths are resolved under this directory;
            otherwise files go to ``outputs/reports/`` (see :data:`visualization.paths.REPORTS_DIR`).
    """
    from visualization.plots import plot_drawdown, plot_equity_curve, plot_trade_distribution

    base = f"{prefix}_" if prefix else ""
    if output_dir is not None:
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        fp_eq = d / f"{base}equity_curve.png"
        fp_dd = d / f"{base}drawdown.png"
        fp_td = d / f"{base}trade_distribution.png"
    else:
        fp_eq = f"{base}equity_curve.png"
        fp_dd = f"{base}drawdown.png"
        fp_td = f"{base}trade_distribution.png"

    return {
        "equity": plot_equity_curve(result.equity, filepath=fp_eq),
        "drawdown": plot_drawdown(result.equity, filepath=fp_dd),
        "trades": plot_trade_distribution(result.trade_legs, filepath=fp_td),
    }
