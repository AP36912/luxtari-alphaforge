"""
Report figures: **drawing** is matplotlib-only; saves PNGs under ``outputs/reports/`` by default.

Confusion matrix **counts** come from ``sklearn.metrics.confusion_matrix``; the heatmap is plain
``imshow`` (no seaborn/plotly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from visualization.paths import REPORTS_DIR, ensure_reports_dir

# Local type alias for trade PnL sources
TradePnlLike = Sequence[float] | np.ndarray


def _resolve_outpath(filepath: str | Path | None, default_name: str) -> Path:
    if filepath is None:
        ensure_reports_dir()
        return REPORTS_DIR / default_name
    p = Path(filepath)
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    ensure_reports_dir()
    out = REPORTS_DIR / p
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _close_fig(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_equity_curve(
    equity: np.ndarray | Sequence[float],
    *,
    x: np.ndarray | Sequence[float] | None = None,
    title: str = "Equity curve",
    ylabel: str = "Equity",
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 120,
) -> Path:
    """Line chart of equity vs bar index (or ``x``)."""
    import matplotlib.pyplot as plt

    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    xv = np.arange(len(eq), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
    if len(xv) != len(eq):
        raise ValueError("x and equity must have the same length")

    out = _resolve_outpath(filepath, "equity_curve.png")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(xv, eq, color="C0", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Bar" if x is None else "Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=dpi)
    _close_fig(fig)
    return out


def plot_drawdown(
    equity: np.ndarray | Sequence[float],
    *,
    x: np.ndarray | Sequence[float] | None = None,
    title: str = "Drawdown",
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (10, 3),
    dpi: int = 120,
) -> Path:
    """Underwater curve (negative fraction from running peak)."""
    import matplotlib.pyplot as plt

    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    if len(eq) < 2:
        raise ValueError("equity must have at least 2 points")
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, np.nan)
    dd = np.nan_to_num(dd, nan=0.0)
    xv = np.arange(len(eq), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
    if len(xv) != len(eq):
        raise ValueError("x and equity must have the same length")

    out = _resolve_outpath(filepath, "drawdown.png")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.fill_between(xv, dd, 0.0, color="C3", alpha=0.35)
    ax.plot(xv, dd, color="C3", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Bar" if x is None else "Index")
    ax.set_ylabel("Drawdown (fraction)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=dpi)
    _close_fig(fig)
    return out


def plot_feature_importance(
    importances: np.ndarray | Sequence[float],
    *,
    feature_names: Sequence[str] | None = None,
    title: str = "Feature importance",
    top_n: int | None = 20,
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> Path:
    """Horizontal bar chart of importances (e.g. tree model ``feature_importances_``)."""
    import matplotlib.pyplot as plt

    imp = np.asarray(importances, dtype=np.float64).reshape(-1)
    n = imp.shape[0]
    if feature_names is not None:
        names = list(feature_names)
        if len(names) != n:
            raise ValueError("feature_names length must match importances")
    else:
        names = [f"f{i}" for i in range(n)]

    order = np.argsort(imp)
    if top_n is not None:
        order = order[-int(top_n) :]
    imp_s = imp[order]
    names_s = [names[i] for i in order]

    out = _resolve_outpath(filepath, "feature_importance.png")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    y_pos = np.arange(len(imp_s))
    ax.barh(y_pos, imp_s, color="C2", alpha=0.85)
    ax.set_yticks(y_pos, labels=names_s, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out, dpi=dpi)
    _close_fig(fig)
    return out


def plot_confusion_matrix(
    y_true: np.ndarray | Sequence,
    y_pred: np.ndarray | Sequence,
    *,
    labels: np.ndarray | Sequence | None = None,
    title: str = "Confusion matrix",
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 120,
) -> Path:
    """Heatmap of class counts (``sklearn.metrics.confusion_matrix`` for the grid)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")

    lab = None if labels is None else np.asarray(labels).reshape(-1)
    cm = confusion_matrix(yt, yp, labels=lab)
    from sklearn.utils.multiclass import unique_labels

    if lab is not None:
        tick_names = [str(x) for x in lab]
    else:
        tick_names = [str(x) for x in unique_labels(yt, yp)]

    out = _resolve_outpath(filepath, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    if len(tick_names) == cm.shape[1] == cm.shape[0]:
        ax.set_xticklabels(tick_names, rotation=45, ha="right")
        ax.set_yticklabels(tick_names)
    else:
        ax.set_xticklabels([str(j) for j in range(cm.shape[1])], rotation=45, ha="right")
        ax.set_yticklabels([str(i) for i in range(cm.shape[0])])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.savefig(out, dpi=dpi)
    _close_fig(fig)
    return out


def _pnl_array(pnls: TradePnlLike | Sequence[object]) -> np.ndarray:
    """Accept raw PnL array/sequence or objects with ``.pnl`` (e.g. ``TradeLeg``)."""
    if isinstance(pnls, np.ndarray):
        return np.asarray(pnls, dtype=np.float64).reshape(-1)
    if not pnls:
        return np.array([], dtype=np.float64)
    first = pnls[0]  # type: ignore[index]
    if hasattr(first, "pnl"):
        return np.array([float(getattr(t, "pnl")) for t in pnls], dtype=np.float64)  # type: ignore[arg-type]
    return np.asarray(pnls, dtype=np.float64).reshape(-1)


def plot_trade_distribution(
    pnls: TradePnlLike | Sequence[object],
    *,
    title: str = "Trade PnL distribution",
    bins: int = 30,
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (8, 4),
    dpi: int = 120,
) -> Path:
    """Histogram of per-trade realized PnL (array/sequence or ``TradeLeg``-like)."""
    import matplotlib.pyplot as plt

    p = _pnl_array(pnls)
    out = _resolve_outpath(filepath, "trade_distribution.png")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    if p.size > 0:
        ax.hist(p, bins=min(bins, max(5, p.size)), color="C1", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Realized PnL")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out, dpi=dpi)
    _close_fig(fig)
    return out
