"""
Write experiment artifacts under ``outputs/reports/experiments/<run_id>/``.

Mirrors key run outputs for a stable research archive path (in addition to ``outputs/runs/``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config.settings import get_project_paths
from reporting.experiment_report import write_experiment_markdown


def experiment_reports_root() -> Path:
    return get_project_paths().outputs_dir / "reports" / "experiments"


def write_experiment_bundle(
    *,
    run_id: str,
    run_dir: Path,
    reports_dir: Path,
    metrics_payload: dict[str, Any],
    audit_findings: list[dict[str, Any]],
    feature_list: list[str],
    label_distribution: dict[str, int],
    model_parameters: dict[str, Any],
    config_snapshot: dict[str, Any],
    metadata: dict[str, Any],
    label_spec_dict: dict[str, Any],
    model_name: str,
    resolved_data_path: str,
    outputs_dir: Path | None = None,
) -> Path:
    """
    Create ``outputs/reports/experiments/<run_id>/`` with JSON summaries, REPORT.md, and PNG copies.

    Args:
        outputs_dir: Project ``outputs/`` root (defaults to :func:`get_project_paths`; pass explicitly
            when tests patch project paths on the pipeline only).
    """
    from utils.paths import confined_output_subdir

    base = outputs_dir if outputs_dir is not None else get_project_paths().outputs_dir
    bundle = confined_output_subdir(base, "reports", "experiments", run_id=run_id)
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "metrics_summary.json").write_text(
        json.dumps(metrics_payload, indent=2, default=str),
        encoding="utf-8",
    )
    (bundle / "feature_list.json").write_text(
        json.dumps(feature_list, indent=2),
        encoding="utf-8",
    )
    (bundle / "label_distribution.json").write_text(
        json.dumps(label_distribution, indent=2),
        encoding="utf-8",
    )
    (bundle / "model_parameters.json").write_text(
        json.dumps(model_parameters, indent=2, default=str),
        encoding="utf-8",
    )
    (bundle / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str),
        encoding="utf-8",
    )
    (bundle / "audit_findings.json").write_text(
        json.dumps(audit_findings, indent=2),
        encoding="utf-8",
    )

    write_experiment_markdown(
        bundle / "REPORT.md",
        run_id=run_id,
        metrics=metrics_payload,
        metadata=metadata,
        label_spec=label_spec_dict,
        model_name=model_name,
        feature_list=feature_list,
        label_distribution=label_distribution,
        resolved_data_path=resolved_data_path,
        audit_findings=audit_findings,
    )

    if reports_dir.is_dir():
        for p in reports_dir.glob("*.png"):
            shutil.copy2(p, bundle / p.name)

    # Optional: copy metrics from run_dir if user looks only at bundle
    m_src = run_dir / "metrics.json"
    if m_src.is_file():
        shutil.copy2(m_src, bundle / "metrics.json")

    return bundle
