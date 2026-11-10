"""
SQLite-backed experiment store (single ``experiments`` table + indexes for leaderboard queries).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    asset TEXT,
    bar_type TEXT,
    timeframe TEXT,
    date_start TEXT,
    date_end TEXT,
    feature_set_json TEXT,
    labeling_json TEXT,
    regime_json TEXT,
    model_type TEXT NOT NULL,
    model_hyperparameters_json TEXT,
    validation_method TEXT,
    validation_config_json TEXT,
    cv_accuracy_mean REAL,
    cv_precision_macro_mean REAL,
    holdout_accuracy REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    profit_factor REAL,
    total_return REAL,
    n_trades INTEGER,
    config_hash TEXT,
    run_dir TEXT,
    metrics_path TEXT,
    model_dir TEXT,
    reports_dir TEXT,
    bundle_dir TEXT,
    audit_path TEXT,
    search_batch_id TEXT,
    search_trial_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_exp_sharpe ON experiments(sharpe);
CREATE INDEX IF NOT EXISTS idx_exp_sortino ON experiments(sortino);
CREATE INDEX IF NOT EXISTS idx_exp_cv_acc ON experiments(cv_accuracy_mean);
CREATE INDEX IF NOT EXISTS idx_exp_batch ON experiments(search_batch_id);
"""


@dataclass
class ExperimentRow:
    """Row payload for :meth:`ExperimentStore.insert_experiment`."""

    experiment_id: str
    created_at: str
    asset: str | None
    bar_type: str | None
    timeframe: str | None
    date_start: str | None
    date_end: str | None
    feature_set_json: str
    labeling_json: str
    regime_json: str
    model_type: str
    model_hyperparameters_json: str
    validation_method: str
    validation_config_json: str
    cv_accuracy_mean: float | None
    cv_precision_macro_mean: float | None
    holdout_accuracy: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    profit_factor: float | None
    total_return: float | None
    n_trades: int | None
    config_hash: str | None
    run_dir: str | None
    metrics_path: str | None
    model_dir: str | None
    reports_dir: str | None
    bundle_dir: str | None
    audit_path: str | None
    search_batch_id: str | None
    search_trial_index: int | None


_INSERT_SQL = """
INSERT OR REPLACE INTO experiments (
    experiment_id, created_at, asset, bar_type, timeframe, date_start, date_end,
    feature_set_json, labeling_json, regime_json,
    model_type, model_hyperparameters_json,
    validation_method, validation_config_json,
    cv_accuracy_mean, cv_precision_macro_mean, holdout_accuracy,
    sharpe, sortino, max_drawdown, profit_factor, total_return, n_trades,
    config_hash, run_dir, metrics_path, model_dir, reports_dir, bundle_dir, audit_path,
    search_batch_id, search_trial_index
) VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""


class ExperimentStore:
    """Thin SQLite API for experiment logging and reads."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(DDL)

    def insert_experiment(self, row: ExperimentRow) -> None:
        self.init_schema()
        vals = (
            row.experiment_id,
            row.created_at,
            row.asset,
            row.bar_type,
            row.timeframe,
            row.date_start,
            row.date_end,
            row.feature_set_json,
            row.labeling_json,
            row.regime_json,
            row.model_type,
            row.model_hyperparameters_json,
            row.validation_method,
            row.validation_config_json,
            row.cv_accuracy_mean,
            row.cv_precision_macro_mean,
            row.holdout_accuracy,
            row.sharpe,
            row.sortino,
            row.max_drawdown,
            row.profit_factor,
            row.total_return,
            row.n_trades,
            row.config_hash,
            row.run_dir,
            row.metrics_path,
            row.model_dir,
            row.reports_dir,
            row.bundle_dir,
            row.audit_path,
            row.search_batch_id,
            row.search_trial_index,
        )
        with self._connect() as conn:
            conn.execute(_INSERT_SQL, vals)
            conn.commit()

    def fetch_by_id(self, experiment_id: str) -> dict[str, Any] | None:
        self.init_schema()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            r = cur.fetchone()
            if r is None:
                return None
            return dict(r)

    def fetch_all_dicts(self, *, where: str = "", params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        self.init_schema()
        sql = "SELECT * FROM experiments"
        if where.strip():
            sql += f" WHERE {where}"
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def row_count(self) -> int:
        self.init_schema()
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
