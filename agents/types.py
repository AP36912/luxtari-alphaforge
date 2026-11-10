"""
Structured inputs and outputs for the supervised research agent layer.

All public payloads are JSON-friendly except where noted (e.g. optional DataFrame handles).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from config.experiment_schema import ExperimentConfig
from models.config import ModelConfig


@dataclass
class ResearchObjective:
    """High-level goal for a supervised research run."""

    description: str
    """Free-text objective (logged and echoed in reports)."""
    experiment: ExperimentConfig
    """Full experiment specification (data, features, labels, validation, backtest)."""
    extra_model_configs: tuple[ModelConfig, ...] = ()
    """Additional models to compare in the modeling agent (beyond ``experiment.model``)."""


@dataclass
class ExperimentPlan:
    """Deterministic step list produced by the supervisor."""

    objective_summary: str
    steps: tuple[str, ...]
    notes: str = ""


@dataclass
class DataAgentOutput:
    schema_columns: tuple[str, ...]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    n_rows: int
    time_start: str | None
    time_end: str | None
    duplicate_timestamps: int
    inferred_bar_type: str
    """``time_regular``, ``time_irregular``, or ``unknown`` (no timestamp column)."""
    median_bar_delta_seconds: float | None
    source_path: str


@dataclass
class FeatureAgentOutput:
    feature_names: tuple[str, ...]
    n_rows: int
    n_dropped_na: int
    feature_config_effective: dict[str, Any]


@dataclass
class ModelComparisonRow:
    model_name: str
    model_type: str
    holdout_accuracy: float
    holdout_precision_macro: float
    holdout_recall_macro: float
    roc_auc_ovo: float | None


@dataclass
class ModelingAgentOutput:
    comparisons: tuple[ModelComparisonRow, ...]
    chosen_model_name: str
    chosen_model_config: ModelConfig


@dataclass
class ValidationAgentOutput:
    cv_accuracy_mean: float
    cv_precision_macro_mean: float
    cv_recall_macro_mean: float
    n_folds: int
    fold_details: tuple[dict[str, Any], ...]
    leakage_risk_messages: tuple[str, ...]
    holdout_oos_accuracy: float
    holdout_oos_precision_macro: float
    holdout_oos_recall_macro: float


@dataclass
class BacktestAgentOutput:
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    total_return: float | None
    n_trades: int


@dataclass
class ResearchWriterOutput:
    markdown_notes: str
    experiment_summary: str


@dataclass
class SupervisorResult:
    """End-to-end agent run (serializable except ``chosen_model_config`` in dict export)."""

    objective: str
    plan: ExperimentPlan
    data: DataAgentOutput
    features: FeatureAgentOutput
    modeling: ModelingAgentOutput
    validation: ValidationAgentOutput
    backtest: BacktestAgentOutput
    writer: ResearchWriterOutput
    final_summary: str
    """Short plain-text conclusion for CLI / logs."""

    def to_dict(self) -> dict[str, Any]:
        """Nested dicts for tests and optional JSON export."""
        return asdict(self)


# --- Step results carrying DataFrames for orchestration (not in SupervisorResult) ---


@dataclass
class DataAgentStepResult:
    output: DataAgentOutput
    ohlcv: pd.DataFrame


@dataclass
class FeatureAgentStepResult:
    output: FeatureAgentOutput
    featured: pd.DataFrame


@dataclass
class LabeledFeatureStepResult:
    """After labeling; consumed by modeling / validation / backtest agents."""

    merged: pd.DataFrame
    X: Any  # np.ndarray — avoid top-level numpy import cycle in typing only
    y: Any
    feature_names: tuple[str, ...]
    split: int
    horizon: int
