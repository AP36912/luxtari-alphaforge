"""
Side-by-side comparison of a small set of logged experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tracking.store import ExperimentStore


_COMPARE_COLUMNS = (
    "experiment_id",
    "created_at",
    "model_type",
    "cv_accuracy_mean",
    "holdout_accuracy",
    "sharpe",
    "sortino",
    "max_drawdown",
    "profit_factor",
    "total_return",
    "n_trades",
    "config_hash",
    "search_batch_id",
    "search_trial_index",
)


def compare_experiments(
    db_path: str | Path,
    experiment_ids: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load rows by ``experiment_id`` and return a :class:`pandas.DataFrame`.

    Missing ids are omitted (no error).
    """
    cols = tuple(columns) if columns else _COMPARE_COLUMNS
    store = ExperimentStore(db_path)
    store.init_schema()
    rows: list[dict[str, Any]] = []
    for eid in experiment_ids:
        r = store.fetch_by_id(str(eid))
        if r:
            rows.append({c: r.get(c) for c in cols})
    if not rows:
        return pd.DataFrame(columns=list(cols))
    return pd.DataFrame(rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """GitHub-style table without optional ``tabulate`` dependency."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([head, sep] + body_lines) + "\n"


def compare_experiments_markdown(
    db_path: str | Path,
    experiment_ids: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Markdown table for notebooks / reports."""
    df = compare_experiments(db_path, experiment_ids, columns=columns)
    if df.empty:
        return "_No matching experiments._\n"
    return _dataframe_to_markdown(df)
