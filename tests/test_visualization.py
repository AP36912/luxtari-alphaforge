"""Visualization saves PNGs (non-interactive Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from backtest.config import BacktestConfig, PositionSizing
from backtest.engine import BacktestEngine, BacktestResult
from backtest.execution import TradeLeg
from visualization.paths import REPORTS_DIR, ensure_reports_dir
from visualization.plots import (
    plot_confusion_matrix,
    plot_drawdown,
    plot_equity_curve,
    plot_feature_importance,
    plot_trade_distribution,
)
from visualization.reports import save_backtest_report


def test_plot_equity_curve_writes_file(tmp_path) -> None:
    p = tmp_path / "eq.png"
    out = plot_equity_curve(np.array([100.0, 101.0, 99.5, 102.0]), filepath=p)
    assert out.resolve() == p.resolve()
    assert p.stat().st_size > 500


def test_plot_drawdown_writes_file(tmp_path) -> None:
    p = tmp_path / "dd.png"
    eq = np.array([100.0, 110.0, 105.0, 120.0])
    out = plot_drawdown(eq, filepath=p)
    assert out == p.resolve()
    assert p.stat().st_size > 500


def test_plot_feature_importance_writes_file(tmp_path) -> None:
    p = tmp_path / "fi.png"
    imp = np.array([0.1, 0.4, 0.05, 0.25])
    names = ["a", "b", "c", "d"]
    out = plot_feature_importance(imp, feature_names=names, top_n=3, filepath=p)
    assert out == p.resolve()
    assert p.stat().st_size > 500


def test_plot_confusion_matrix_writes_file(tmp_path) -> None:
    p = tmp_path / "cm.png"
    yt = np.array([0, 1, 1, 0, 2])
    yp = np.array([0, 1, 0, 0, 2])
    out = plot_confusion_matrix(yt, yp, labels=np.array([0, 1, 2]), filepath=p)
    assert out == p.resolve()
    assert p.stat().st_size > 800


def test_plot_trade_distribution_trade_legs(tmp_path) -> None:
    p = tmp_path / "td.png"
    legs = [TradeLeg(bar=1, pnl=10.0), TradeLeg(bar=2, pnl=-3.0), TradeLeg(bar=3, pnl=5.0)]
    out = plot_trade_distribution(legs, filepath=p)
    assert out == p.resolve()
    assert p.stat().st_size > 500


def test_default_reports_dir_created() -> None:
    ensure_reports_dir()
    assert REPORTS_DIR.is_dir()


def test_save_backtest_report(tmp_path) -> None:
    eq = np.linspace(100_000, 101_000, 20)
    res = BacktestResult(
        equity=eq,
        cash=np.zeros_like(eq),
        position=np.zeros_like(eq),
        open=np.ones_like(eq),
        close=np.ones_like(eq),
        signals=np.zeros_like(eq),
        trade_legs=[TradeLeg(bar=2, pnl=50.0), TradeLeg(bar=5, pnl=-10.0)],
        config=BacktestConfig(),
    )
    paths = save_backtest_report(res, prefix="run", output_dir=tmp_path)
    assert len(paths) == 3
    for _k, path in paths.items():
        assert path.is_file()
        assert path.stat().st_size > 400


def test_save_backtest_report_integration_minimal() -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100, 101, 101],
            "low": [100, 100, 101],
            "close": [100.0, 100.0, 101.0],
        }
    )
    sig = np.array([0.0, 1.0, 0.0])
    res = BacktestEngine(
        BacktestConfig(initial_capital=10_000.0, sizing=PositionSizing(fixed_shares=1.0))
    ).run(df, sig)
    paths = save_backtest_report(res, prefix="pytest_bt")
    for p in paths.values():
        assert p.is_file()
        p.unlink(missing_ok=True)
