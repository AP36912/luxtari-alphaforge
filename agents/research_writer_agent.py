"""
Research writer agent: deterministic markdown from structured agent outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.types import (
    BacktestAgentOutput,
    DataAgentOutput,
    ExperimentPlan,
    FeatureAgentOutput,
    ModelingAgentOutput,
    ResearchWriterOutput,
    ValidationAgentOutput,
)


@dataclass
class ResearchWriterInput:
    objective: str
    plan: ExperimentPlan
    data: DataAgentOutput
    features: FeatureAgentOutput
    modeling: ModelingAgentOutput
    validation: ValidationAgentOutput
    backtest: BacktestAgentOutput


def _fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def run_research_writer_agent(inp: ResearchWriterInput) -> ResearchWriterOutput:
    """Emit markdown research notes plus a short plain-text experiment summary."""
    lines: list[str] = [
        "# Supervised research run",
        "",
        "## Objective",
        "",
        inp.objective.strip() or "(none)",
        "",
        "## Experiment plan",
        "",
    ]
    for s in inp.plan.steps:
        lines.append(f"- {s}")
    if inp.plan.notes.strip():
        lines.extend(["", f"*Notes:* {inp.plan.notes.strip()}", ""])

    lines.extend(
        [
            "## Data profile",
            "",
            f"- **Path:** `{inp.data.source_path}`",
            f"- **Rows:** {inp.data.n_rows}",
            f"- **Time range:** {inp.data.time_start} → {inp.data.time_end}",
            f"- **Bar inference:** {inp.data.inferred_bar_type}"
            + (
                f" (median Δ ≈ {inp.data.median_bar_delta_seconds:.3g}s)"
                if inp.data.median_bar_delta_seconds is not None
                else ""
            ),
            f"- **Duplicate timestamps:** {inp.data.duplicate_timestamps}",
            "- **Missing (core columns):** "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(inp.data.missing_counts.items()))
                if inp.data.missing_counts
                else "none"
            ),
            "",
            "## Features",
            "",
            f"- **Count:** {len(inp.features.feature_names)}",
            f"- **Rows after pipeline:** {inp.features.n_rows}",
            f"- **Rows dropped (NaN):** {inp.features.n_dropped_na}",
            "",
            "## Model comparison (holdout)",
            "",
        ]
    )
    for row in inp.modeling.comparisons:
        lines.append(
            f"- **{row.model_name}:** acc={row.holdout_accuracy:.4f}, "
            f"P_macro={row.holdout_precision_macro:.4f}, R_macro={row.holdout_recall_macro:.4f}"
            + (f", ROC AUC={row.roc_auc_ovo:.4f}" if row.roc_auc_ovo is not None else "")
        )
    lines.extend(
        [
            "",
            f"- **Chosen:** `{inp.modeling.chosen_model_name}`",
            "",
            "## Validation",
            "",
            f"- **Purged CV mean accuracy:** {inp.validation.cv_accuracy_mean:.4f}",
            f"- **Purged CV mean precision (macro):** {inp.validation.cv_precision_macro_mean:.4f}",
            f"- **Purged CV mean recall (macro):** {inp.validation.cv_recall_macro_mean:.4f}",
            f"- **Holdout OOS accuracy:** {inp.validation.holdout_oos_accuracy:.4f}",
            "",
            "### Leakage / risk messages",
            "",
        ]
    )
    if not inp.validation.leakage_risk_messages:
        lines.append("*No leakage warnings or errors from heuristics.*")
    else:
        for msg in inp.validation.leakage_risk_messages:
            lines.append(f"- {msg}")
    lines.extend(
        [
            "",
            "## Backtest (holdout)",
            "",
            f"- **Sharpe:** {_fmt(inp.backtest.sharpe)}",
            f"- **Sortino:** {_fmt(inp.backtest.sortino)}",
            f"- **Max drawdown:** {_fmt(inp.backtest.max_drawdown)}",
            f"- **Total return:** {_fmt(inp.backtest.total_return)}",
            f"- **Trades:** {inp.backtest.n_trades}",
            "",
        ]
    )

    markdown_notes = "\n".join(lines) + "\n"
    experiment_summary = (
        f"Objective: {inp.objective.strip()[:200]!r}; "
        f"chosen_model={inp.modeling.chosen_model_name}; "
        f"cv_acc_mean={inp.validation.cv_accuracy_mean:.4f}; "
        f"holdout_acc={inp.validation.holdout_oos_accuracy:.4f}; "
        f"sharpe={_fmt(inp.backtest.sharpe)}; "
        f"n_trades={inp.backtest.n_trades}"
    )
    return ResearchWriterOutput(markdown_notes=markdown_notes, experiment_summary=experiment_summary)
