"""
Markdown summaries for logged search batches (best run, robust run, top trials, sensitivity).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracking.leaderboard import best_run, most_robust_run
from tracking.store import ExperimentStore


def _loads(s: str | None) -> Any:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _model_hp_sensitivity(rows: list[dict[str, Any]]) -> str:
    hp_keys: set[str] = set()
    for r in rows:
        hp_keys |= set(_loads(r.get("model_hyperparameters_json")).keys())
    blocks: list[str] = []
    for k in sorted(hp_keys):
        by_val: dict[Any, list[float | None]] = defaultdict(list)
        for r in rows:
            hp = _loads(r.get("model_hyperparameters_json"))
            if k not in hp:
                continue
            sh = r.get("sharpe")
            by_val[hp[k]].append(float(sh) if sh is not None else None)
        if len(by_val) < 2:
            continue
        lines = [f"### Model hyperparameter `{k}`", ""]
        for vk in sorted(by_val.keys(), key=lambda x: (str(type(x)), str(x))):
            xs = [x for x in by_val[vk] if x is not None]
            mean_s = sum(xs) / len(xs) if xs else 0.0
            lines.append(f"- **{vk!r}:** mean Sharpe ≈ {mean_s:.4f} (trials={len(by_val[vk])})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "*No multi-value model hyperparameters in batch.*"


def _labeling_sensitivity(rows: list[dict[str, Any]]) -> str:
    tb_keys: set[str] = set()
    for r in rows:
        lab = _loads(r.get("labeling_json"))
        tb = lab.get("triple_barrier") or {}
        if isinstance(tb, dict):
            tb_keys |= set(tb.keys())
    blocks: list[str] = []
    for k in sorted(tb_keys):
        by_val: dict[Any, list[float | None]] = defaultdict(list)
        for r in rows:
            lab = _loads(r.get("labeling_json"))
            tb = lab.get("triple_barrier") or {}
            if not isinstance(tb, dict) or k not in tb:
                continue
            sh = r.get("sharpe")
            by_val[tb[k]].append(float(sh) if sh is not None else None)
        if len(by_val) < 2:
            continue
        lines = [f"### Labeling `triple_barrier.{k}`", ""]
        for vk in sorted(by_val.keys(), key=lambda x: (str(type(x)), str(x))):
            xs = [x for x in by_val[vk] if x is not None]
            mean_s = sum(xs) / len(xs) if xs else 0.0
            lines.append(f"- **{vk!r}:** mean Sharpe ≈ {mean_s:.4f} (trials={len(by_val[vk])})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else ""


def summarize_search_batch(
    db_path: str | Path,
    batch_id: str,
    *,
    top_n: int = 8,
    robust_min_cv: float = 0.45,
) -> str:
    """
    Build a markdown report: best Sharpe, most robust (CV floor), top trials table, sensitivity notes.
    """
    store = ExperimentStore(db_path)
    store.init_schema()
    rows = store.fetch_all_dicts(where="search_batch_id = ?", params=(batch_id,))
    if not rows:
        return f"# Search batch `{batch_id}`\n\n_No experiments logged for this batch._\n"

    lines = [
        f"# Search batch summary: `{batch_id}`",
        "",
        f"- **Trials:** {len(rows)}",
        "",
        "## Best run (Sharpe)",
        "",
    ]
    b = best_run(db_path, sort_by="sharpe", search_batch_id=batch_id)
    if b:
        lines.append(f"- **experiment_id:** `{b.get('experiment_id')}`")
        lines.append(f"- **Sharpe:** {b.get('sharpe')}")
        lines.append(f"- **CV acc mean:** {b.get('cv_accuracy_mean')}")
        lines.append(f"- **model:** {b.get('model_type')} `{b.get('model_hyperparameters_json')}`")
    else:
        lines.append("*No finite Sharpe values.*")
    lines.extend(["", "## Most robust run (CV floor + Sharpe)", ""])
    rb = most_robust_run(
        db_path,
        search_batch_id=batch_id,
        min_cv_accuracy=robust_min_cv,
        sort_by="sharpe",
    )
    if rb:
        lines.append(f"- **experiment_id:** `{rb.get('experiment_id')}`")
        lines.append(f"- **Sharpe:** {rb.get('sharpe')}")
        lines.append(f"- **CV acc mean:** {rb.get('cv_accuracy_mean')}")
    else:
        lines.append(f"_No trial meets cv_accuracy_mean ≥ {robust_min_cv}._")

    lines.extend(["", f"## Top {top_n} by Sharpe", ""])
    sorted_rows = sorted(
        [r for r in rows if r.get("sharpe") is not None],
        key=lambda r: float(r["sharpe"]),
        reverse=True,
    )[:top_n]
    if not sorted_rows:
        lines.append("_No Sharpe data._")
    else:
        lines.append("| id | Sharpe | CV acc | Holdout acc | PF | model hp |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in sorted_rows:
            lines.append(
                f"| `{r.get('experiment_id')}` | {r.get('sharpe')} | {r.get('cv_accuracy_mean')} | "
                f"{r.get('holdout_accuracy')} | {r.get('profit_factor')} | `{r.get('model_hyperparameters_json')}` |"
            )

    lines.extend(["", "## Sensitivity (mean Sharpe by parameter value)", ""])
    lines.append(_model_hp_sensitivity(rows))
    lab_s = _labeling_sensitivity(rows)
    if lab_s:
        lines.extend(["", lab_s])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rankings use **holdout** backtest Sharpe unless noted; purged CV stats are for generalization screening.",
            "- **Robust** picks a simple CV floor then Sharpe — tune ``robust_min_cv`` for your label noise.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
