"""
Rank logged experiments by metric (Sharpe, CV accuracy, drawdown, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from tracking.store import ExperimentStore

SortKey = Literal[
    "sharpe",
    "sortino",
    "max_drawdown",
    "profit_factor",
    "cv_accuracy_mean",
    "holdout_accuracy",
    "total_return",
]


def leaderboard(
    db_path: str | Path,
    *,
    sort_by: SortKey = "sharpe",
    ascending: bool = False,
    limit: int = 20,
    search_batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return top ``limit`` rows ordered by ``sort_by``.

    For ``max_drawdown``, ascending=True means **less negative first** (smaller loss).
    """
    col = sort_by
    if col not in (
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "cv_accuracy_mean",
        "holdout_accuracy",
        "total_return",
    ):
        raise ValueError(f"unsupported sort_by: {sort_by!r}")

    store = ExperimentStore(db_path)
    where_parts: list[str] = [f"{col} IS NOT NULL"]
    params: list[Any] = []
    if search_batch_id is not None:
        where_parts.append("search_batch_id = ?")
        params.append(search_batch_id)
    where_sql = " AND ".join(where_parts)
    order = "ASC" if ascending else "DESC"
    # NULLS LAST behavior: sqlite sorts NULL first in ASC — filter IS NOT NULL already
    sql = f"""
        SELECT * FROM experiments
        WHERE {where_sql}
        ORDER BY {col} {order}
        LIMIT ?
    """
    params.append(int(limit))
    store.init_schema()
    with store._connect() as conn:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def best_run(
    db_path: str | Path,
    *,
    sort_by: SortKey = "sharpe",
    search_batch_id: str | None = None,
) -> dict[str, Any] | None:
    rows = leaderboard(db_path, sort_by=sort_by, limit=1, search_batch_id=search_batch_id)
    return rows[0] if rows else None


def most_robust_run(
    db_path: str | Path,
    *,
    search_batch_id: str | None,
    min_cv_accuracy: float = 0.45,
    sort_by: SortKey = "sharpe",
) -> dict[str, Any] | None:
    """
    Among runs with ``cv_accuracy_mean >= min_cv_accuracy``, pick best ``sort_by``.

    Simple proxy for "not overfit to backtest only".
    """
    store = ExperimentStore(db_path)
    store.init_schema()
    where = "cv_accuracy_mean IS NOT NULL AND cv_accuracy_mean >= ?"
    params: list[Any] = [min_cv_accuracy]
    if search_batch_id is not None:
        where += " AND search_batch_id = ?"
        params.append(search_batch_id)
    order = "DESC" if sort_by != "max_drawdown" else "ASC"
    col = sort_by
    sql = f"""
        SELECT * FROM experiments
        WHERE {where}
        ORDER BY {col} {order}
        LIMIT 1
    """
    with store._connect() as conn:
        cur = conn.execute(sql, params)
        r = cur.fetchone()
        return dict(r) if r else None
