"""Research charts (matplotlib); defaults to ``outputs/reports/``."""

from visualization.paths import REPORTS_DIR, ensure_reports_dir, project_root
from visualization.plots import (
    plot_confusion_matrix,
    plot_drawdown,
    plot_equity_curve,
    plot_feature_importance,
    plot_trade_distribution,
)
from visualization.reports import save_backtest_report

__all__ = [
    "REPORTS_DIR",
    "ensure_reports_dir",
    "plot_confusion_matrix",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_feature_importance",
    "plot_trade_distribution",
    "project_root",
    "save_backtest_report",
]
