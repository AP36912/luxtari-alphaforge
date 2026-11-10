"""
Markdown research summary for one experiment run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt_num(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def write_experiment_markdown(
    out_path: Path,
    *,
    run_id: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    label_spec: dict[str, Any],
    model_name: str,
    feature_list: list[str],
    label_distribution: dict[str, int],
    resolved_data_path: str,
    audit_findings: list[dict[str, Any]],
) -> None:
    """Write a human-readable experiment report."""
    asset = metadata.get("asset", "unspecified")
    timeframe = metadata.get("timeframe", "unspecified")
    bar_type = metadata.get("bar_type", "unspecified")
    notes = metadata.get("notes", "")

    cv = metrics.get("cv", {})
    ho = metrics.get("holdout_classification", {})
    bt = metrics.get("holdout_backtest", {})

    lines: list[str] = [
        f"# Experiment report: `{run_id}`",
        "",
        "## Summary",
        "",
        f"- **Experiment ID:** `{run_id}`",
        f"- **Config hash:** `{metrics.get('config_hash', 'n/a')}`",
        f"- **Seed:** `{metrics.get('seed', 'n/a')}`",
        f"- **Rows used:** `{metrics.get('n_rows_used', 'n/a')}`",
        f"- **Label horizon (bars):** `{metrics.get('label_horizon', 'n/a')}`",
        "",
        "## Dataset",
        "",
        f"- **Asset:** {asset}",
        f"- **Timeframe / bar type:** {timeframe} / {bar_type}",
        f"- **Resolved data path:** `{resolved_data_path}`",
        "",
        "## Features",
        "",
        f"- **Count:** {len(feature_list)}",
        "- **Names (first 40):**",
        "",
    ]
    for name in feature_list[:40]:
        lines.append(f"  - `{name}`")
    if len(feature_list) > 40:
        lines.append(f"  - … *({len(feature_list) - 40} more)*")
    lines.extend(["", "## Labeling", "", "```yaml"])
    for k, v in sorted(label_spec.items(), key=lambda kv: str(kv[0])):
        lines.append(f"{k}: {v}")
    lines.extend(["```", "", "## Label distribution (training sample)", ""])
    for k, v in sorted(label_distribution.items(), key=lambda kv: str(kv[0])):
        lines.append(f"- **{k}:** {v}")
    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- **Name:** `{model_name}`",
            "",
            "## Validation (Purged K-Fold)",
            "",
            f"- **Folds:** {cv.get('n_folds', 'n/a')}",
            f"- **Mean accuracy:** {_fmt_num(cv.get('accuracy_mean'))}",
            f"- **Mean precision (macro):** {_fmt_num(cv.get('precision_macro_mean'))}",
            f"- **Mean recall (macro):** {_fmt_num(cv.get('recall_macro_mean'))}",
            "",
            "## Holdout classification",
            "",
            f"- **Train / test rows:** {ho.get('n_train', 'n/a')} / {ho.get('n_test', 'n/a')}",
            f"- **Accuracy:** {_fmt_num(ho.get('accuracy'))}",
            f"- **Precision (macro):** {_fmt_num(ho.get('precision_macro'))}",
            f"- **Recall (macro):** {_fmt_num(ho.get('recall_macro'))}",
            f"- **ROC AUC (ovo):** {_fmt_num(ho.get('roc_auc_ovo'))}",
            "",
            "## Backtest (holdout, engine contract)",
            "",
            f"- **Sharpe:** {_fmt_num(bt.get('sharpe'))}",
            f"- **Sortino:** {_fmt_num(bt.get('sortino'))}",
            f"- **Max drawdown:** {_fmt_num(bt.get('max_drawdown'))}",
            f"- **Win rate:** {_fmt_num(bt.get('win_rate'))}",
            f"- **Profit factor:** {_fmt_num(bt.get('profit_factor'))}",
            f"- **Total return:** {_fmt_num(bt.get('total_return'))}",
            f"- **Trade legs:** {_fmt_num(bt.get('n_trades'))}",
            "",
        ]
    )
    seg = metrics.get("holdout_backtest_segments") or {}
    if isinstance(seg, dict) and seg:
        lines.extend(
            [
                "## Backtest segments (session / regime)",
                "",
                "Subset metrics use masked bar returns and trades whose ``bar`` index lies in the segment.",
                "",
            ]
        )
        for name in sorted(seg.keys()):
            d = seg[name]
            if not isinstance(d, dict):
                continue
            lines.append(
                f"- **{name}:** Sharpe={_fmt_num(d.get('sharpe'))}, "
                f"trades={_fmt_num(d.get('n_trades'))}, "
                f"bars={_fmt_num(d.get('n_bars_in_mask'))}"
            )
        lines.append("")
    lines.extend(
        [
            "## Audit findings",
            "",
        ]
    )
    if not audit_findings:
        lines.append("*No findings recorded.*")
    else:
        for f in audit_findings:
            sev = f.get("severity", "?")
            code = f.get("code", "")
            msg = f.get("message", "")
            lines.append(f"- **[{sev}]** `{code}` — {msg}")
    lines.extend(
        [
            "",
            "## Top risks / caveats",
            "",
            "- **Holdout backtest** uses model predictions on the **holdout slice only**; signals align row-wise with that slice’s OHLC (next-bar open fills).",
            "- **Purged K-fold** may place training samples after the test block by design; do not confuse with a simple chronological split.",
            "- **Leakage heuristics** (names + next-return correlation) catch *obvious* issues only — they do not certify causal correctness of every feature.",
            "- **Costs and slippage** in backtest are stylized; live execution may differ.",
            "- **Class imbalance** and **low trade counts** can make Sharpe / profit factor unstable; check `label_distribution` and `n_trades`.",
        ]
    )
    if notes:
        lines.extend(["", "## Run notes (from config metadata)", "", str(notes), ""])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
