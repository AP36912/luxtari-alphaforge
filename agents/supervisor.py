"""
Supervisor agent: plan a deterministic research workflow and orchestrate sub-agents.

Uses existing framework modules for labeling, models, validation, and backtests — no duplicate
business logic beyond glue and ordering.
"""

from __future__ import annotations

import numpy as np

from agents.backtest_agent import BacktestAgentInput, run_backtest_agent
from agents.data_agent import DataAgentInput, run_data_agent
from agents.feature_agent import FeatureAgentInput, run_feature_agent
from agents.modeling_agent import ModelingAgentInput, run_modeling_agent
from agents.research_writer_agent import ResearchWriterInput, run_research_writer_agent
from agents.types import (
    ExperimentPlan,
    LabeledFeatureStepResult,
    ResearchObjective,
    SupervisorResult,
)
from agents.utils import compute_holdout_split_index, resolve_data_path, seed_model_params
from agents.validation_agent import ValidationAgentInput, run_validation_agent
from config.experiment_schema import backtest_config_from_dict, triple_barrier_config_from_dict
from config.pipeline import set_global_seeds
from features.pipeline import feature_column_names
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels
from models.config import ModelConfig
from models.registry import create_model
from models.training import ModelTrainer


def build_experiment_plan(obj: ResearchObjective) -> ExperimentPlan:
    """Define a fixed step sequence from the research objective (no LLM)."""
    n_extra = len(obj.extra_model_configs)
    notes = (
        f"Registry primary model: `{obj.experiment.model.name}`; "
        f"{n_extra} additional model(s) in comparison set."
    )
    steps = (
        "data_agent — load CSV, schema / missingness / time range / bar regularity",
        "feature_agent — causal feature pipeline (FeatureConfig)",
        "labeling — triple-barrier targets (labeling.triple_barrier)",
        "modeling_agent — chronological split, compare models, pick winner (deterministic tie-break)",
        "validation_agent — purged K-fold + leakage heuristics for chosen model",
        "backtest_agent — holdout OHLC + chosen-model predictions",
        "research_writer_agent — markdown notes and executive summary",
    )
    return ExperimentPlan(
        objective_summary=obj.description.strip(),
        steps=steps,
        notes=notes,
    )


def _prepare_labeled_matrix(obj: ResearchObjective, featured) -> LabeledFeatureStepResult:
    import pandas as pd

    if not isinstance(featured, pd.DataFrame):
        raise TypeError("featured frame must be a DataFrame")
    tb = triple_barrier_config_from_dict(obj.experiment.labeling)
    horizon = tb.max_bars
    lab_df = compute_triple_barrier_labels(featured, tb)
    merged = merge_labels(featured, lab_df)
    y = merged["triple_barrier_label"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    merged = merged.loc[valid].reset_index(drop=True)
    y = merged["triple_barrier_label"].to_numpy(dtype=np.float64)
    feat_names = tuple(feature_column_names(featured))
    if not feat_names:
        raise ValueError("No feature columns after pipeline; check FeatureConfig")
    X = merged[list(feat_names)].to_numpy(dtype=np.float64)
    n = len(merged)
    if n < obj.experiment.validation.n_splits + 10:
        raise ValueError(f"Not enough rows ({n}) for validation.n_splits")

    split = compute_holdout_split_index(
        n,
        horizon=horizon,
        validation=obj.experiment.validation,
    )
    return LabeledFeatureStepResult(
        merged=merged,
        X=X,
        y=y,
        feature_names=feat_names,
        split=split,
        horizon=horizon,
    )


def _holdout_predictions(
    X: np.ndarray,
    y: np.ndarray,
    split: int,
    model_config: ModelConfig,
    seed: int,
) -> np.ndarray:
    extra = seed_model_params(dict(model_config.extra_params), seed)
    mconf = ModelConfig(name=model_config.name, extra_params=extra)
    model = create_model(mconf)
    trainer = ModelTrainer(model)
    trainer.fit(X[:split], y[:split])
    return np.asarray(trainer.predict(X[split:]), dtype=np.float64)


def run_supervised_research(obj: ResearchObjective) -> SupervisorResult:
    """
    Execute the full agent chain for one :class:`ResearchObjective`.

    Order: data → features → labeling (internal) → modeling → validation → backtest → writer.
    """
    cfg = obj.experiment
    set_global_seeds(cfg.seed)

    plan = build_experiment_plan(obj)

    data_path = resolve_data_path(cfg)
    data_step = run_data_agent(DataAgentInput(path=str(data_path), source=cfg.data.source))

    feat_step = run_feature_agent(
        FeatureAgentInput(ohlcv=data_step.ohlcv, feature_overrides=dict(cfg.features))
    )

    labeled = _prepare_labeled_matrix(obj, feat_step.featured)
    labels_for_metrics = np.unique(labeled.y).astype(int).tolist()

    model_configs = (cfg.model,) + tuple(obj.extra_model_configs)
    modeling = run_modeling_agent(
        ModelingAgentInput(
            X=labeled.X,
            y=labeled.y,
            split=labeled.split,
            model_configs=model_configs,
            base_seed=cfg.seed,
            labels_for_metrics=labels_for_metrics,
        )
    )

    validation = run_validation_agent(
        ValidationAgentInput(
            merged=labeled.merged,
            feature_names=labeled.feature_names,
            X=labeled.X,
            y=labeled.y,
            split=labeled.split,
            horizon=labeled.horizon,
            validation=cfg.validation,
            model_config=modeling.chosen_model_config,
            base_seed=cfg.seed,
        )
    )

    pred_te = _holdout_predictions(
        labeled.X,
        labeled.y,
        labeled.split,
        modeling.chosen_model_config,
        cfg.seed + 999,
    )
    merged_te = labeled.merged.iloc[labeled.split :].reset_index(drop=True)
    ohlc_te = merged_te[["open", "high", "low", "close"]]
    bt_cfg = backtest_config_from_dict(cfg.backtest)
    backtest = run_backtest_agent(
        BacktestAgentInput(ohlc_holdout=ohlc_te, signals=pred_te, backtest_config=bt_cfg)
    )

    writer_in = ResearchWriterInput(
        objective=obj.description,
        plan=plan,
        data=data_step.output,
        features=feat_step.output,
        modeling=modeling,
        validation=validation,
        backtest=backtest,
    )
    writer = run_research_writer_agent(writer_in)

    final_summary = writer.experiment_summary

    return SupervisorResult(
        objective=obj.description,
        plan=plan,
        data=data_step.output,
        features=feat_step.output,
        modeling=modeling,
        validation=validation,
        backtest=backtest,
        writer=writer,
        final_summary=final_summary,
    )
