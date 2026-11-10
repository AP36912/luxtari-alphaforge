"""
Compare backtest metrics (from ``metrics.json``) with paper session summaries (Phase 16.5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("quant_ai.reporting.compare")


@dataclass
class ComparisonPaths:
    backtest_metrics_json: Path
    paper_summary_json: Path
    output_dir: Path


def assumptions_match(a: dict[str, Any] | None, b: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if a is None or b is None:
        return False, ["missing_assumptions_block"]
    diffs: list[str] = []
    for k in ("fill_basis", "stop_target_fill_policy"):
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)!r} vs {b.get(k)!r}")
    ca = a.get("costs") or {}
    cb = b.get("costs") or {}
    for ck in ("half_spread", "slippage", "commission_rate"):
        if float(ca.get(ck, 0)) != float(cb.get(ck, 0)):
            diffs.append(f"costs.{ck}")
    return len(diffs) == 0, diffs


def generate_comparison_report(
    *,
    backtest_metrics: dict[str, Any],
    paper_summary: dict[str, Any],
    output_dir: Path,
    label: str = "compare",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hb = backtest_metrics.get("holdout_backtest") or {}
    pa = paper_summary.get("assumptions")
    pb = backtest_metrics.get("resolved_backtest_assumptions") or backtest_metrics.get(
        "backtest_execution"
    )
    match, diffs = assumptions_match(pa, pb if isinstance(pb, dict) else None)

    lines = [
        "# Backtest vs paper comparison",
        "",
        f"**Assumptions match:** {'yes' if match else 'no — see diffs'}",
        "",
    ]
    if not match:
        lines.extend(["## Assumption diffs", ""] + [f"- {d}" for d in diffs] + [""])

    lines.extend(
        [
            "## Side-by-side",
            "",
            "| Metric | Backtest holdout | Paper |",
            "| --- | --- | --- |",
            f"| fill_basis | {(pb or {}).get('fill_basis') if isinstance(pb, dict) else '—'} | {(pa or {}).get('fill_basis') if isinstance(pa, dict) else '—'} |",
            f"| n_trades | {hb.get('n_trades')} | {paper_summary.get('n_trades')} |",
            f"| win_rate | {hb.get('win_rate')} | {paper_summary.get('win_rate')} |",
            f"| total_pnl (approx) | {hb.get('total_return')} | {paper_summary.get('total_pnl')} |",
            f"| max_drawdown | {hb.get('max_drawdown')} | {paper_summary.get('max_drawdown')} |",
            "",
            f"*Paper date range:* {paper_summary.get('date_start')} → {paper_summary.get('date_end')}",
            "",
        ]
    )

    md_path = output_dir / f"{label}_backtest_paper.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "assumptions_aligned": match,
        "assumption_diffs": diffs,
        "backtest_holdout": hb,
        "paper": {k: paper_summary.get(k) for k in paper_summary if k != "assumptions"},
        "paper_assumptions": pa,
        "backtest_assumptions_snapshot": pb,
    }
    js_path = output_dir / f"{label}_backtest_paper.json"
    js_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("comparison_report_generated markdown=%s json=%s", md_path, js_path)
    return md_path, js_path


def run_compare_cli(
    metrics_path: Path,
    paper_path: Path,
    out_dir: Path,
    label: str = "run",
) -> tuple[Path, Path]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    if "resolved_backtest_assumptions" not in metrics:
        cr_path = metrics_path.parent / "config_resolved.json"
        if cr_path.is_file():
            cr = json.loads(cr_path.read_text(encoding="utf-8"))
            bt = cr.get("backtest") or {}
            from config.experiment_schema import execution_assumptions_from_backtest_dict

            metrics["resolved_backtest_assumptions"] = execution_assumptions_from_backtest_dict(
                bt
            ).to_dict()
    return generate_comparison_report(
        backtest_metrics=metrics,
        paper_summary=paper,
        output_dir=out_dir,
        label=label,
    )
