"""Build :class:`ExperimentRow` from a completed pipeline run."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

from config.experiment_schema import ExperimentConfig, TrackingConfig
from tracking.store import ExperimentRow, iso_now


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _finite_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def build_experiment_row(
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    result_paths: dict[str, Any],
    *,
    data_date_start: str | None,
    data_date_end: str | None,
    tracking: TrackingConfig,
) -> ExperimentRow:
    """Flatten config + metrics into a SQLite row."""
    meta = dict(cfg.metadata)
    cv = metrics.get("cv") or {}
    ho = metrics.get("holdout_classification") or {}
    bt = metrics.get("holdout_backtest") or {}

    return ExperimentRow(
        experiment_id=str(metrics.get("run_id", result_paths.get("run_id", "unknown"))),
        created_at=iso_now(),
        asset=str(meta.get("asset")) if meta.get("asset") is not None else None,
        bar_type=str(meta.get("bar_type")) if meta.get("bar_type") is not None else None,
        timeframe=str(meta.get("timeframe")) if meta.get("timeframe") is not None else None,
        date_start=data_date_start,
        date_end=data_date_end,
        feature_set_json=_json_dumps(cfg.features),
        labeling_json=_json_dumps(cfg.labeling),
        regime_json=_json_dumps(dataclasses.asdict(cfg.regime)),
        model_type=str(cfg.model.name),
        model_hyperparameters_json=_json_dumps(dict(cfg.model.extra_params)),
        validation_method="purged_kfold",
        validation_config_json=_json_dumps(dataclasses.asdict(cfg.validation)),
        cv_accuracy_mean=_finite_or_none(cv.get("accuracy_mean")),
        cv_precision_macro_mean=_finite_or_none(cv.get("precision_macro_mean")),
        holdout_accuracy=_finite_or_none(ho.get("accuracy")),
        sharpe=_finite_or_none(bt.get("sharpe")),
        sortino=_finite_or_none(bt.get("sortino")),
        max_drawdown=_finite_or_none(bt.get("max_drawdown")),
        profit_factor=_finite_or_none(bt.get("profit_factor")),
        total_return=_finite_or_none(bt.get("total_return")),
        n_trades=int(bt["n_trades"]) if bt.get("n_trades") is not None else None,
        config_hash=str(metrics.get("config_hash")) if metrics.get("config_hash") else None,
        run_dir=str(result_paths.get("run_dir", "")) or None,
        metrics_path=str(result_paths.get("metrics_path", "")) or None,
        model_dir=str(result_paths.get("model_dir", "")) or None,
        reports_dir=str(result_paths.get("reports_dir", "")) or None,
        bundle_dir=str(result_paths.get("experiment_reports_bundle", "")) or None,
        audit_path=str(result_paths.get("audit_path", "")) or None,
        search_batch_id=tracking.search_batch_id,
        search_trial_index=int(tracking.search_trial_index)
        if tracking.search_trial_index is not None
        else None,
    )


def log_experiment_run(
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    result_paths: dict[str, Any],
    *,
    data_date_start: str | None,
    data_date_end: str | None,
    db_path_resolved: str,
) -> None:
    """Persist one run if ``cfg.tracking.enabled``."""
    if not cfg.tracking.enabled:
        return
    row = build_experiment_row(
        cfg,
        metrics,
        result_paths,
        data_date_start=data_date_start,
        data_date_end=data_date_end,
        tracking=cfg.tracking,
    )
    from tracking.store import ExperimentStore

    ExperimentStore(db_path_resolved).insert_experiment(row)
