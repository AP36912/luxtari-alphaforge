"""Tests for the supervised agent layer (deterministic orchestration)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.data_agent import DataAgentInput, run_data_agent
from agents.feature_agent import FeatureAgentInput, run_feature_agent
from agents.supervisor import build_experiment_plan, run_supervised_research
from agents.types import ResearchObjective
from agents.utils import compute_holdout_split_index
from config.experiment_schema import DataConfig, ExperimentConfig, OutputConfig, ValidationConfig
from config.settings import ProjectPaths
from models.config import ModelConfig


def _synthetic_ohlcv(n: int = 400, *, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz=None)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0.0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.2, n)
    vol = rng.uniform(100.0, 1000.0, n)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def _fake_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(root=root, data_dir=root / "data", outputs_dir=root / "outputs")


def test_data_agent_profiles_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "p"
    root.mkdir()
    monkeypatch.setattr("agents.utils.get_project_paths", lambda: _fake_paths(root))

    csv_path = tmp_path / "bars.csv"
    _synthetic_ohlcv(120).to_csv(csv_path, index=False)

    res = run_data_agent(DataAgentInput(path=str(csv_path.resolve())))
    assert res.output.n_rows == 120
    assert res.output.inferred_bar_type == "time_regular"
    assert "timestamp" in res.output.dtypes
    assert len(res.ohlcv) == 120


def test_feature_agent_builds_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "p2"
    root.mkdir()
    monkeypatch.setattr("agents.utils.get_project_paths", lambda: _fake_paths(root))

    csv_path = tmp_path / "b2.csv"
    _synthetic_ohlcv(200).to_csv(csv_path, index=False)
    data = run_data_agent(DataAgentInput(path=str(csv_path.resolve())))
    feat = run_feature_agent(FeatureAgentInput(ohlcv=data.ohlcv, feature_overrides={}))
    assert len(feat.output.feature_names) > 0
    assert feat.output.n_rows <= data.output.n_rows


def test_supervisor_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr("agents.utils.get_project_paths", lambda: _fake_paths(root))

    csv_path = tmp_path / "e2e.csv"
    _synthetic_ohlcv(450, seed=42).to_csv(csv_path, index=False)

    cfg = ExperimentConfig(
        seed=11,
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
            extra_params={"n_estimators": 25, "max_depth": 4, "n_jobs": 1},
        ),
        validation=ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.72),
        backtest={
            "initial_capital": 50_000.0,
            "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
            "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
        },
        output=OutputConfig(run_id=None),
    )
    extra = (
        ModelConfig(
            name="random_forest",
            extra_params={"n_estimators": 15, "max_depth": 3, "n_jobs": 1},
        ),
    )
    obj = ResearchObjective(
        description="Smoke test supervised research",
        experiment=cfg,
        extra_model_configs=extra,
    )
    plan = build_experiment_plan(obj)
    assert len(plan.steps) == 7

    result = run_supervised_research(obj)
    assert result.modeling.chosen_model_name == "random_forest"
    assert result.validation.n_folds == 3
    assert "Supervised research" in result.writer.markdown_notes
    assert "random_forest" in result.final_summary
    d = result.to_dict()
    assert d["objective"] == "Smoke test supervised research"
    assert d["backtest"]["n_trades"] >= 0


def test_compute_holdout_split_index_matches_fraction_band() -> None:
    vc = ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.7)
    s = compute_holdout_split_index(500, horizon=5, validation=vc)
    assert 200 < s < 480
