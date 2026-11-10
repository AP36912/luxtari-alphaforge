"""
End-to-end experiment: data → features → labels → train → CV → holdout → backtest → artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit.backtest_alignment import audit_backtest_inputs
from audit.leakage import audit_correlation_vs_next_return, audit_feature_names
from audit.repro import assert_numpy_determinism_after_seed, environment_snapshot
from audit.results import AuditReport, audit_findings_as_jsonable
from audit.splits import audit_holdout_prefix_split, audit_purged_kfold_fold
from audit.timestamps import audit_timestamps
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.metrics import summarize
from backtest.segmentation import compare_backtest_segments
from config.experiment_schema import (
    ExperimentConfig,
    backtest_config_from_dict,
    feature_config_from_overrides,
    triple_barrier_config_from_dict,
)
from config.settings import get_project_paths
from data.loaders.csv_loader import load_csv
from features.pipeline import build_features, feature_column_names
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels
from models.config import ModelConfig
from models.registry import create_model
from models.persistence import read_meta
from models.training import ModelTrainer
from regime.filters import apply_signal_gate, trading_allowed_mask
from regime.integration import attach_regime_columns, regime_feature_column_names
from regime.masks import build_holdout_segment_masks
from validation.metrics import accuracy, precision_recall, roc_auc
from validation.purged_kfold import PurgedKFold
from visualization.plots import plot_confusion_matrix, plot_feature_importance
from visualization.reports import save_backtest_report

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """Best-effort reproducibility across numpy / Python / torch."""
    import os

    os.environ["PYTHONHASHSEED"] = str(seed % (2**31 - 1))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def experiment_fingerprint_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """JSON-serializable spec for hashing (no paths)."""
    return {
        "seed": cfg.seed,
        "data": dataclasses.asdict(cfg.data),
        "features": cfg.features,
        "labeling": cfg.labeling,
        "model": {"name": cfg.model.name, "extra_params": cfg.model.extra_params},
        "validation": dataclasses.asdict(cfg.validation),
        "walk_forward_calibration": dataclasses.asdict(cfg.walk_forward_calibration),
        "backtest": cfg.backtest,
        "output": dataclasses.asdict(cfg.output),
        "metadata": dict(cfg.metadata),
        "audit": dataclasses.asdict(cfg.audit),
        "regime": dataclasses.asdict(cfg.regime),
        "tracking": dataclasses.asdict(cfg.tracking),
        "live_ops": dataclasses.asdict(cfg.live_ops),
        "execution": dataclasses.asdict(cfg.execution),
    }


def _label_distribution(y: np.ndarray) -> dict[str, int]:
    uniq, counts = np.unique(y, return_counts=True)
    return {str(float(u)): int(c) for u, c in zip(uniq, counts)}


def _collect_model_parameters(model_dir: Path, trainer: ModelTrainer) -> dict[str, Any]:
    payload: dict[str, Any] = {"meta": read_meta(model_dir)}
    est = getattr(trainer.model, "_estimator", None)
    if est is not None and hasattr(est, "get_params"):
        try:
            payload["sklearn_get_params"] = est.get_params(deep=True)
        except Exception:  # noqa: BLE001
            payload["sklearn_get_params"] = None
    return payload


def _log_audit_report(rep: AuditReport) -> None:
    for f in rep.findings:
        if f.severity == "error":
            logger.error("Audit [%s] %s", f.code, f.message)
        elif f.severity == "warning":
            logger.warning("Audit [%s] %s", f.code, f.message)
        else:
            logger.info("Audit [%s] %s", f.code, f.message)


def _strict_gate(rep: AuditReport, strict: bool) -> None:
    if strict:
        rep.raise_for_errors()


def config_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(experiment_fingerprint_dict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def resolve_data_path(cfg: ExperimentConfig) -> Path:
    paths = get_project_paths()
    p = Path(cfg.data.path)
    if not str(cfg.data.path).strip():
        raise ValueError("data.path is required")
    return p if p.is_absolute() else paths.root / p


def load_ohlcv_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    path = resolve_data_path(cfg)
    if cfg.data.source != "csv":
        raise ValueError(f"Unsupported data.source: {cfg.data.source!r}")
    logger.info("Loading OHLCV from %s", path)
    return load_csv(path)


def _confined_run_dir(outputs_dir: Path, run_id: str) -> Path:
    """Resolve a run directory and reject absolute/traversal output escapes."""
    from utils.paths import confined_output_subdir

    return confined_output_subdir(outputs_dir, "runs", run_id=run_id)


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Execute the full pipeline and write artifacts under ``outputs/runs/<run_id>/``.

    Returns a dict with paths and metric summaries (also written to ``metrics.json``).
    """
    paths = get_project_paths()
    run_id = cfg.output.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _confined_run_dir(paths.outputs_dir, run_id)
    model_dir = run_dir / "model"
    reports_dir = run_dir / "reports"
    for d in (run_dir, model_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_global_seeds(cfg.seed)
    fp = config_hash(cfg)
    logger.info("Experiment run_id=%s config_hash=%s", run_id, fp)

    audit_report = AuditReport()
    acfg = cfg.audit
    if acfg.check_numpy_seed_sanity and not assert_numpy_determinism_after_seed(cfg.seed):
        audit_report.add(
            "warning",
            "REPRO_SEED",
            "numpy RNG sanity check failed (identical re-seed did not reproduce draw)",
        )

    labeling_cfg = dict(cfg.labeling or {})
    label_mode = str(labeling_cfg.get("mode") or "triple_barrier")
    label_spec_dict: dict[str, Any]
    strategy_diag: dict[str, Any] | None = None
    tb = None
    if label_mode != "triple_barrier":
        raise ValueError(f"unsupported public labeling.mode: {label_mode!r}")
    tb = triple_barrier_config_from_dict(labeling_cfg)
    horizon = int(tb.max_bars)
    label_spec_dict = {**dataclasses.asdict(tb), "mode": "triple_barrier"}
    if cfg.validation.label_horizon != horizon:
        logger.warning(
            "validation.label_horizon (%s) != labeling triple_barrier.max_bars (%s); using max_bars for PurgedKFold",
            cfg.validation.label_horizon,
            horizon,
        )

    raw = load_ohlcv_frame(cfg)
    if acfg.check_timestamps and "timestamp" in raw.columns:
        audit_report.extend(audit_timestamps(raw))
    _strict_gate(audit_report, acfg.strict)

    feat_cfg = feature_config_from_overrides(cfg.features)
    logger.info("Building features")
    featured = build_features(raw, feat_cfg, dropna=True, include_ohlcv=True)

    logger.info("Computing labels (mode=%s)", label_mode)
    assert tb is not None
    lab_df = compute_triple_barrier_labels(featured, tb)
    merged = merge_labels(featured, lab_df)

    y = merged["triple_barrier_label"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    merged = merged.loc[valid].reset_index(drop=True)
    y = merged["triple_barrier_label"].to_numpy(dtype=np.float64)

    n_lab = len(y)
    n_neg = int((y == -1).sum())
    n_zero = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    logger.info(
        "Label distribution (finite labels, mode=%s, n=%s): "
        "-1=%s (%.2f%%), 0=%s (%.2f%%), 1=%s (%.2f%%)",
        label_mode,
        n_lab,
        n_neg,
        100.0 * n_neg / n_lab,
        n_zero,
        100.0 * n_zero / n_lab,
        n_pos,
        100.0 * n_pos / n_lab,
    )

    feat_names_base = feature_column_names(featured)
    if not feat_names_base:
        raise ValueError("No feature columns after pipeline; check FeatureConfig")

    n = len(merged)
    if n < cfg.validation.n_splits + 10:
        raise ValueError(f"Not enough rows ({n}) for n_splits={cfg.validation.n_splits}")

    split = int(cfg.validation.train_fraction * n)
    split = max(split, 2 * horizon + cfg.validation.n_splits + 5)
    split = min(split, n - max(20, horizon + 5))
    if split >= n - 5:
        raise ValueError("train_fraction leaves too few holdout rows")

    r_cfg = cfg.regime
    if r_cfg.enabled:
        logger.info("Attaching regime/session columns (method=%s)", r_cfg.detection_method)
        merged = attach_regime_columns(merged, r_cfg, fit_end_index=split)

    feat_names = feat_names_base + regime_feature_column_names(r_cfg)
    X = merged[feat_names].to_numpy(dtype=np.float64)

    if acfg.check_timestamps and "timestamp" in merged.columns:
        audit_report.extend(audit_timestamps(merged))
    if acfg.check_leakage:
        audit_report.extend(audit_feature_names(feat_names))
        audit_report.extend(
            audit_correlation_vs_next_return(
                merged,
                feat_names,
                max_abs_corr=acfg.max_abs_corr_vs_next_return,
            )
        )
    _strict_gate(audit_report, acfg.strict)

    labels_for_metrics = np.unique(y).astype(int).tolist()

    # --- Purged K-Fold CV ---
    pkf = PurgedKFold(
        cfg.validation.n_splits,
        label_horizon=horizon,
        embargo=cfg.validation.embargo,
    )
    fold_rows: list[dict[str, Any]] = []
    logger.info("Purged K-fold CV (%s splits)", cfg.validation.n_splits)
    for fold_i, (tr_idx, te_idx) in enumerate(pkf.split(X)):
        if acfg.check_splits:
            audit_report.extend(audit_purged_kfold_fold(tr_idx, te_idx, n_samples=n))
        mconf = ModelConfig(
            name=cfg.model.name,
            extra_params=_seed_model_params(dict(cfg.model.extra_params), cfg.seed + fold_i),
        )
        model = create_model(mconf)
        trainer = ModelTrainer(model)
        trainer.fit(X[tr_idx], y[tr_idx])
        pred = trainer.predict(X[te_idx])
        proba = None
        if model.supports_proba:
            try:
                proba = trainer.predict_proba(X[te_idx])
            except Exception:  # noqa: BLE001
                proba = None
        row: dict[str, Any] = {
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "accuracy": accuracy(y[te_idx], pred),
        }
        p, r = precision_recall(
            y[te_idx], pred, average="macro", labels=labels_for_metrics, zero_division=0.0
        )
        row["precision_macro"] = p
        row["recall_macro"] = r
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            try:
                row["roc_auc_ovo"] = roc_auc(y[te_idx], proba, labels=labels_for_metrics)
            except Exception:  # noqa: BLE001
                row["roc_auc_ovo"] = None
        else:
            row["roc_auc_ovo"] = None
        fold_rows.append(row)
        logger.info("Fold %s accuracy=%.4f", fold_i, row["accuracy"])

    _strict_gate(audit_report, acfg.strict)

    cv_summary = {
        "n_folds": len(fold_rows),
        "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_rows])),
        "precision_macro_mean": float(np.mean([f["precision_macro"] for f in fold_rows])),
        "recall_macro_mean": float(np.mean([f["recall_macro"] for f in fold_rows])),
        "folds": fold_rows,
    }

    # --- Holdout train → metrics + backtest OOS ---
    if acfg.check_splits:
        audit_report.extend(audit_holdout_prefix_split(split, n))
    _strict_gate(audit_report, acfg.strict)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    merged_te = merged.iloc[split:].reset_index(drop=True)

    logger.info("Training final model on first %s rows; holdout %s rows", split, n - split)
    final_conf = ModelConfig(
        name=cfg.model.name,
        extra_params=_seed_model_params(dict(cfg.model.extra_params), cfg.seed + 999),
    )
    final_model = create_model(final_conf)
    trainer = ModelTrainer(final_model)
    trainer.fit(X_tr, y_tr)
    pred_te = trainer.predict(X_te)
    proba_te = trainer.predict_proba(X_te) if final_model.supports_proba else None

    holdout = {
        "n_train": int(split),
        "n_test": int(n - split),
        "accuracy": accuracy(y_te, pred_te),
    }
    p, r = precision_recall(y_te, pred_te, average="macro", labels=labels_for_metrics, zero_division=0.0)
    holdout["precision_macro"] = p
    holdout["recall_macro"] = r
    if proba_te is not None:
        try:
            holdout["roc_auc_ovo"] = roc_auc(y_te, proba_te, labels=labels_for_metrics)
        except Exception:  # noqa: BLE001
            holdout["roc_auc_ovo"] = None
    else:
        holdout["roc_auc_ovo"] = None

    trainer.save(model_dir)
    logger.info("Saved model to %s", model_dir)

    # Reports: confusion matrix, feature importance (if available)
    plot_confusion_matrix(
        y_te,
        pred_te,
        labels=np.array(labels_for_metrics),
        filepath=reports_dir / "confusion_matrix.png",
        title="Holdout confusion matrix",
    )
    est = getattr(trainer.model, "_estimator", None)
    if est is not None and hasattr(est, "feature_importances_"):
        plot_feature_importance(
            np.asarray(est.feature_importances_),
            feature_names=feat_names,
            top_n=min(30, len(feat_names)),
            filepath=reports_dir / "feature_importance.png",
        )

    # --- Backtest on holdout OHLC with model signals ---
    bt_cfg = backtest_config_from_dict(cfg.backtest)
    signals = np.asarray(pred_te, dtype=np.float64)
    if r_cfg.enabled and r_cfg.filters.enabled and r_cfg.filters.apply_to_signals:
        sess_te = merged_te["regime_session"].values
        vol_te = merged_te["regime_vol_state"].values
        gate = trading_allowed_mask(
            sess_te,
            vol_te,
            allowed_sessions=r_cfg.filters.allowed_sessions if r_cfg.filters.allowed_sessions else None,
            allowed_volatility_states=r_cfg.filters.allowed_volatility_states
            if r_cfg.filters.allowed_volatility_states
            else None,
            require_both=r_cfg.filters.require_both,
        )
        signals = apply_signal_gate(signals, gate)
        logger.info("Applied regime/session signal gate on holdout (%s bars allowed)", int(gate.sum()))

    ohlc_te = merged_te[["open", "high", "low", "close"]]
    if acfg.check_backtest_alignment:
        audit_report.extend(
            audit_backtest_inputs(
                ohlc_te,
                signals,
                predictions_source="holdout_classifier_predictions",
            )
        )
    _strict_gate(audit_report, acfg.strict)

    engine = BacktestEngine(bt_cfg)
    bt_result = engine.run(ohlc_te, signals)
    bt_metrics = summarize(
        bt_result.equity,
        bt_result.trade_legs,
        bars_per_year=bt_cfg.bars_per_year,
    )
    save_backtest_report(bt_result, prefix="holdout", output_dir=reports_dir)

    holdout_bt = {
        "sharpe": bt_metrics.sharpe,
        "sortino": bt_metrics.sortino,
        "max_drawdown": bt_metrics.max_drawdown,
        "win_rate": bt_metrics.win_rate,
        "profit_factor": bt_metrics.profit_factor,
        "total_return": bt_metrics.total_return,
        "n_trades": bt_metrics.n_trades,
    }

    holdout_bt_segments: dict[str, dict[str, Any]] = {}
    if r_cfg.enabled and r_cfg.include_segment_metrics:
        seg_masks = build_holdout_segment_masks(merged_te, r_cfg.filters)
        holdout_bt_segments = compare_backtest_segments(
            bt_result.equity,
            bt_result.trade_legs,
            seg_masks,
            bars_per_year=bt_cfg.bars_per_year,
        )

    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": fp,
        "seed": cfg.seed,
        "n_rows_used": n,
        "label_horizon": horizon,
        "labeling_mode": label_mode,
        "feature_count": len(feat_names),
        "cv": cv_summary,
        "holdout_classification": holdout,
        "holdout_backtest": holdout_bt,
        "holdout_backtest_segments": holdout_bt_segments,
    }
    if strategy_diag is not None:
        metrics_payload["strategy_reversal_diagnostics"] = strategy_diag

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote metrics to %s", metrics_path)

    _log_audit_report(audit_report)
    audit_path = run_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit_findings_as_jsonable(audit_report), indent=2),
        encoding="utf-8",
    )

    # Save resolved config copy for audit
    cfg_copy = experiment_fingerprint_dict(cfg)
    cfg_copy["resolved_data_path"] = str(resolve_data_path(cfg))
    cfg_copy["feature_columns"] = feat_names
    (run_dir / "config_resolved.json").write_text(
        json.dumps(cfg_copy, indent=2, default=str),
        encoding="utf-8",
    )

    resolved_data = str(resolve_data_path(cfg))
    config_snapshot = {
        **experiment_fingerprint_dict(cfg),
        "resolved_data_path": resolved_data,
        "environment": environment_snapshot(),
    }
    model_parameters = _collect_model_parameters(model_dir, trainer)
    label_distribution = _label_distribution(y)

    from reporting.experiment_bundle import write_experiment_bundle

    bundle_dir = write_experiment_bundle(
        run_id=run_id,
        run_dir=run_dir,
        reports_dir=reports_dir,
        metrics_payload=metrics_payload,
        audit_findings=audit_findings_as_jsonable(audit_report),
        feature_list=list(feat_names),
        label_distribution=label_distribution,
        model_parameters=model_parameters,
        config_snapshot=config_snapshot,
        metadata=dict(cfg.metadata),
        label_spec_dict=label_spec_dict,
        model_name=cfg.model.name,
        resolved_data_path=resolved_data,
        outputs_dir=paths.outputs_dir,
    )
    logger.info("Wrote experiment report bundle to %s", bundle_dir)

    data_date_start: str | None = None
    data_date_end: str | None = None
    if "timestamp" in merged.columns and len(merged) > 0:
        ts_col = merged["timestamp"]
        data_date_start = str(ts_col.min())
        data_date_end = str(ts_col.max())

    result: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "reports_dir": reports_dir,
        "metrics_path": metrics_path,
        "metrics": metrics_payload,
        "audit_path": audit_path,
        "experiment_reports_bundle": bundle_dir,
        "data_date_start": data_date_start,
        "data_date_end": data_date_end,
    }

    if cfg.tracking.enabled:
        from tracking.paths import resolve_tracking_db_path
        from tracking.record import log_experiment_run

        log_experiment_run(
            cfg,
            metrics_payload,
            result,
            data_date_start=data_date_start,
            data_date_end=data_date_end,
            db_path_resolved=str(resolve_tracking_db_path(cfg)),
        )

    return result


def _seed_model_params(extra: dict[str, Any], seed: int) -> dict[str, Any]:
    """Force ``random_state`` so each CV fold / final fit is reproducible."""
    out = dict(extra)
    out["random_state"] = int(seed)
    return out


def run_experiment_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML then run :func:`run_experiment`."""
    from config.yaml_loader import load_experiment_yaml

    raw_path = Path(path).resolve()
    cfg = load_experiment_yaml(raw_path)
    return run_experiment(cfg)


def run_experiment_from_template(
    template_id: str,
    *,
    merge_path: str | Path | None = None,
    data_path: str | None = None,
    extra_overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run an experiment from a :mod:`research_templates` id.

    Optional ``merge_path`` YAML is deep-merged on top of the template (same rules
    as ``template:`` in a single YAML file). ``data_path`` sets ``data.path`` when given.
    """
    import copy

    import yaml

    from research_templates.merge import apply_named_template, deep_merge

    overlay: dict[str, Any] = {}
    if merge_path is not None:
        mp = Path(merge_path).resolve()
        with mp.open("r", encoding="utf-8") as f:
            frag = yaml.safe_load(f) or {}
        if not isinstance(frag, dict):
            raise ValueError("merge YAML root must be a mapping")
        overlay = copy.deepcopy(frag)
    if extra_overlay:
        overlay = deep_merge(overlay, dict(extra_overlay))
    if data_path is not None:
        overlay.setdefault("data", {})["path"] = str(data_path)
    raw = apply_named_template(template_id, overlay)
    from config.experiment_schema import experiment_config_from_dict

    src = Path(merge_path).resolve() if merge_path else Path(f"template:{template_id}")
    cfg = experiment_config_from_dict(raw, source_path=src)
    return run_experiment(cfg)
