"""Experiment store, hyperparameter grid search, leaderboard."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.experiment_schema import (
    DataConfig,
    ExperimentConfig,
    OutputConfig,
    TrackingConfig,
    ValidationConfig,
)
from config.pipeline import run_experiment
from config.settings import ProjectPaths
from models.config import ModelConfig
from reporting.tracking_summary import summarize_search_batch
from tracking.apply_params import apply_param_path
from tracking.compare import compare_experiments
from tracking.leaderboard import leaderboard
from tracking.paths import resolve_tracking_db_path
from tracking.record import build_experiment_row
from tracking.search import run_parameter_search
from tracking.store import ExperimentRow, ExperimentStore


def _fake_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(root=root, data_dir=root / "data", outputs_dir=root / "outputs")


def _synth_csv(path: Path, n: int = 380) -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-06-01", periods=n, freq="h", tz=None)
    c = 100.0 + np.cumsum(rng.normal(0, 0.35, n))
    o = np.r_[c[0], c[:-1]]
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": o,
            "high": np.maximum(o, c) + 0.1,
            "low": np.minimum(o, c) - 0.1,
            "close": c,
            "volume": rng.uniform(50, 200, n),
        }
    ).to_csv(path, index=False)


def test_experiment_store_roundtrip_integrity(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    store = ExperimentStore(db)
    row = ExperimentRow(
        experiment_id="unit_1",
        created_at="2026-01-01T00:00:00+00:00",
        asset="XAU",
        bar_type="time",
        timeframe="1h",
        date_start="2021-01-01",
        date_end="2021-02-01",
        feature_set_json="{}",
        labeling_json="{}",
        regime_json="{}",
        model_type="random_forest",
        model_hyperparameters_json='{"n_estimators": 10}',
        validation_method="purged_kfold",
        validation_config_json="{}",
        cv_accuracy_mean=0.55,
        cv_precision_macro_mean=0.5,
        holdout_accuracy=0.52,
        sharpe=0.8,
        sortino=0.9,
        max_drawdown=-0.1,
        profit_factor=1.2,
        total_return=0.03,
        n_trades=5,
        config_hash="abc",
        run_dir="/r",
        metrics_path="/m.json",
        model_dir="/mod",
        reports_dir="/rep",
        bundle_dir="/b",
        audit_path="/a.json",
        search_batch_id="batch_x",
        search_trial_index=0,
    )
    store.insert_experiment(row)
    got = store.fetch_by_id("unit_1")
    assert got is not None
    assert got["experiment_id"] == "unit_1"
    assert got["sharpe"] == 0.8
    assert got["asset"] == "XAU"
    assert json.loads(got["model_hyperparameters_json"])["n_estimators"] == 10
    assert store.row_count() == 1


def test_build_experiment_row_from_metrics() -> None:
    cfg = ExperimentConfig(
        metadata={"asset": "TEST", "bar_type": "time", "timeframe": "1h"},
        model=ModelConfig(name="random_forest", extra_params={"n_estimators": 5}),
    )
    cfg = dataclasses.replace(
        cfg,
        tracking=TrackingConfig(enabled=True, search_batch_id="b", search_trial_index=2),
    )
    metrics = {
        "run_id": "rid",
        "config_hash": "h1",
        "cv": {"accuracy_mean": 0.6, "precision_macro_mean": 0.55},
        "holdout_classification": {"accuracy": 0.58},
        "holdout_backtest": {
            "sharpe": 0.5,
            "sortino": 0.4,
            "max_drawdown": -0.05,
            "profit_factor": float("inf"),
            "total_return": 0.01,
            "n_trades": 3,
        },
    }
    paths = {"run_dir": "/x", "metrics_path": "/m", "model_dir": "/mo", "reports_dir": "/r", "experiment_reports_bundle": "/bu", "audit_path": "/au"}
    row = build_experiment_row(
        cfg,
        metrics,
        paths,
        data_date_start="2020-01-01",
        data_date_end="2020-06-01",
        tracking=cfg.tracking,
    )
    assert row.experiment_id == "rid"
    assert row.profit_factor is None
    assert row.search_batch_id == "b"
    assert row.search_trial_index == 2


def test_apply_param_path_model_and_labeling() -> None:
    cfg = ExperimentConfig()
    c2 = apply_param_path(cfg, "model.extra_params.n_estimators", 77)
    assert c2.model.extra_params["n_estimators"] == 77
    c3 = apply_param_path(cfg, "labeling.triple_barrier.max_bars", 9)
    assert c3.labeling["triple_barrier"]["max_bars"] == 9
    c4 = apply_param_path(cfg, "features.rsi_period", 21)
    assert c4.features["rsi_period"] == 21


def test_compare_experiments_side_by_side(tmp_path: Path) -> None:
    db = tmp_path / "cmp.db"
    store = ExperimentStore(db)
    for eid, sh in [("a", 1.0), ("b", 0.5)]:
        store.insert_experiment(
            ExperimentRow(
                experiment_id=eid,
                created_at="2026-01-01T00:00:00+00:00",
                asset=None,
                bar_type=None,
                timeframe=None,
                date_start=None,
                date_end=None,
                feature_set_json="{}",
                labeling_json="{}",
                regime_json="{}",
                model_type="rf",
                model_hyperparameters_json="{}",
                validation_method="purged_kfold",
                validation_config_json="{}",
                cv_accuracy_mean=None,
                cv_precision_macro_mean=None,
                holdout_accuracy=None,
                sharpe=sh,
                sortino=None,
                max_drawdown=None,
                profit_factor=None,
                total_return=None,
                n_trades=None,
                config_hash=None,
                run_dir=None,
                metrics_path=None,
                model_dir=None,
                reports_dir=None,
                bundle_dir=None,
                audit_path=None,
                search_batch_id=None,
                search_trial_index=None,
            )
        )
    df = compare_experiments(db, ["a", "b"])
    assert len(df) == 2
    assert set(df["experiment_id"]) == {"a", "b"}


def test_leaderboard_ordering(tmp_path: Path) -> None:
    db = tmp_path / "lb.db"
    store = ExperimentStore(db)
    for i, sh in enumerate([0.5, 2.0, 1.0]):
        store.insert_experiment(
            ExperimentRow(
                experiment_id=f"e{i}",
                created_at="2026-01-01T00:00:00+00:00",
                asset=None,
                bar_type=None,
                timeframe=None,
                date_start=None,
                date_end=None,
                feature_set_json="{}",
                labeling_json="{}",
                regime_json="{}",
                model_type="rf",
                model_hyperparameters_json="{}",
                validation_method="purged_kfold",
                validation_config_json="{}",
                cv_accuracy_mean=0.5,
                cv_precision_macro_mean=None,
                holdout_accuracy=None,
                sharpe=sh,
                sortino=None,
                max_drawdown=None,
                profit_factor=None,
                total_return=None,
                n_trades=None,
                config_hash=None,
                run_dir=None,
                metrics_path=None,
                model_dir=None,
                reports_dir=None,
                bundle_dir=None,
                audit_path=None,
                search_batch_id=None,
                search_trial_index=None,
            )
        )
    top = leaderboard(db, sort_by="sharpe", limit=2, ascending=False)
    assert [r["experiment_id"] for r in top] == ["e1", "e2"]


def test_parameter_search_runs_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "tr"
    root.mkdir()
    monkeypatch.setattr("config.pipeline.get_project_paths", lambda: _fake_paths(root))
    csv_path = tmp_path / "s.csv"
    _synth_csv(csv_path)

    db_path = root / "outputs" / "exp.db"
    base = ExperimentConfig(
        seed=5,
        data=DataConfig(source="csv", path=str(csv_path.resolve())),
        features={},
        labeling={
            "triple_barrier": {
                "max_bars": 5,
                "profit_pct": 0.02,
                "stop_pct": 0.02,
                "use_volatility_scaling": False,
            }
        },
        model=ModelConfig(
            name="random_forest",
            extra_params={"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
        ),
        validation=ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.72),
        backtest={
            "initial_capital": 30_000.0,
            "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
            "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
        },
        output=OutputConfig(run_id=None),
        tracking=TrackingConfig(enabled=False, db_path=str(db_path)),
    )

    batch = "pytest_grid_01"
    results = run_parameter_search(
        base,
        {"model.extra_params.n_estimators": (8, 10)},
        batch_id=batch,
        db_path=db_path,
    )
    assert len(results) == 2
    assert all(r["search_batch_id"] == batch for r in results)
    store = ExperimentStore(db_path)
    batch_rows = store.fetch_all_dicts(where="search_batch_id = ?", params=(batch,))
    assert len(batch_rows) == 2
    ne = {json.loads(r["model_hyperparameters_json"])["n_estimators"] for r in batch_rows}
    assert ne == {8, 10}


def test_summarize_search_batch_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "sm"
    root.mkdir()
    monkeypatch.setattr("config.pipeline.get_project_paths", lambda: _fake_paths(root))
    csv_path = tmp_path / "s2.csv"
    _synth_csv(csv_path, n=360)
    db_path = root / "outputs" / "s.db"

    base = ExperimentConfig(
        seed=1,
        data=DataConfig(source="csv", path=str(csv_path.resolve())),
        features={},
        labeling={
            "triple_barrier": {
                "max_bars": 5,
                "profit_pct": 0.02,
                "stop_pct": 0.02,
                "use_volatility_scaling": False,
            }
        },
        model=ModelConfig(
            name="random_forest",
            extra_params={"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
        ),
        validation=ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.72),
        backtest={
            "initial_capital": 25_000.0,
            "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
            "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
        },
        output=OutputConfig(run_id=None),
        tracking=TrackingConfig(enabled=False, db_path=str(db_path)),
    )
    batch = "pytest_summary_batch"
    run_parameter_search(
        base,
        {"model.extra_params.max_depth": (3, 4)},
        batch_id=batch,
        db_path=db_path,
    )
    md = summarize_search_batch(db_path, batch, top_n=3, robust_min_cv=0.0)
    assert batch in md
    assert "Best run" in md or "best" in md.lower()


def test_resolve_tracking_db_path_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tracking.paths.get_project_paths", lambda: _fake_paths(tmp_path))
    cfg = ExperimentConfig(tracking=TrackingConfig(db_path="outputs/x.db"))
    p = resolve_tracking_db_path(cfg)
    assert p == tmp_path / "outputs" / "x.db"


def test_pipeline_logs_when_tracking_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "plog"
    root.mkdir()
    monkeypatch.setattr("config.pipeline.get_project_paths", lambda: _fake_paths(root))
    csv_path = tmp_path / "pl.csv"
    _synth_csv(csv_path, n=400)
    db_path = root / "outputs" / "track.db"

    cfg = ExperimentConfig(
        seed=9,
        data=DataConfig(source="csv", path=str(csv_path.resolve())),
        features={},
        labeling={
            "triple_barrier": {
                "max_bars": 5,
                "profit_pct": 0.02,
                "stop_pct": 0.02,
                "use_volatility_scaling": False,
            }
        },
        model=ModelConfig(
            name="random_forest",
            extra_params={"n_estimators": 15, "max_depth": 3, "n_jobs": 1},
        ),
        validation=ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.72),
        backtest={
            "initial_capital": 20_000.0,
            "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
            "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
        },
        output=OutputConfig(run_id="tracked_single"),
        metadata={"asset": "XAU", "bar_type": "time"},
        tracking=TrackingConfig(enabled=True, db_path=str(db_path)),
    )
    out = run_experiment(cfg)
    assert out.get("data_date_start")
    got = ExperimentStore(db_path).fetch_by_id("tracked_single")
    assert got is not None
    assert got["asset"] == "XAU"
