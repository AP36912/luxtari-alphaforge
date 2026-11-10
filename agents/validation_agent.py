"""
Validation agent: purged K-fold via ``validation.purged_kfold``, leakage heuristics via ``audit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.types import ValidationAgentOutput
from agents.utils import seed_model_params
from audit.leakage import audit_correlation_vs_next_return, audit_feature_names
from audit.results import AuditReport
from config.experiment_schema import ValidationConfig
from models.config import ModelConfig
from models.registry import create_model
from models.training import ModelTrainer
from validation.metrics import accuracy, precision_recall
from validation.purged_kfold import PurgedKFold


@dataclass
class ValidationAgentInput:
    merged: object  # pd.DataFrame
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    split: int
    horizon: int
    validation: ValidationConfig
    model_config: ModelConfig
    base_seed: int


def _collect_leakage_messages(rep: AuditReport) -> tuple[str, ...]:
    msgs: list[str] = []
    for f in rep.findings:
        if f.severity in ("warning", "error"):
            msgs.append(f"[{f.code}] {f.message}")
    return tuple(msgs)


def run_validation_agent(inp: ValidationAgentInput) -> ValidationAgentOutput:
    """Run leakage screens on the labeled frame and purged CV for the given model."""
    import pandas as pd

    merged = inp.merged
    if not isinstance(merged, pd.DataFrame):
        raise TypeError("merged must be a pandas DataFrame")

    leak_rep = AuditReport()
    leak_rep.extend(audit_feature_names(inp.feature_names))
    leak_rep.extend(
        audit_correlation_vs_next_return(
            merged,
            list(inp.feature_names),
            max_abs_corr=0.995,
        )
    )
    leakage_msgs = _collect_leakage_messages(leak_rep)

    labels = np.unique(inp.y).astype(int).tolist()
    pkf = PurgedKFold(
        inp.validation.n_splits,
        label_horizon=inp.horizon,
        embargo=inp.validation.embargo,
    )
    fold_rows: list[dict[str, object]] = []
    for fold_i, (tr_idx, te_idx) in enumerate(pkf.split(inp.X)):
        extra = seed_model_params(dict(inp.model_config.extra_params), inp.base_seed + fold_i)
        mconf = ModelConfig(name=inp.model_config.name, extra_params=extra)
        model = create_model(mconf)
        trainer = ModelTrainer(model)
        trainer.fit(inp.X[tr_idx], inp.y[tr_idx])
        pred = trainer.predict(inp.X[te_idx])
        row: dict[str, object] = {
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "accuracy": accuracy(inp.y[te_idx], pred),
        }
        p, r = precision_recall(
            inp.y[te_idx], pred, average="macro", labels=labels, zero_division=0.0
        )
        row["precision_macro"] = p
        row["recall_macro"] = r
        fold_rows.append(row)

    acc_m = float(np.mean([float(f["accuracy"]) for f in fold_rows]))
    p_m = float(np.mean([float(f["precision_macro"]) for f in fold_rows]))
    r_m = float(np.mean([float(f["recall_macro"]) for f in fold_rows]))

    # Holdout OOS (same chronological split as modeling agent)
    split = inp.split
    extra = seed_model_params(dict(inp.model_config.extra_params), inp.base_seed + 9999)
    mconf = ModelConfig(name=inp.model_config.name, extra_params=extra)
    model = create_model(mconf)
    trainer = ModelTrainer(model)
    trainer.fit(inp.X[:split], inp.y[:split])
    pred_te = trainer.predict(inp.X[split:])
    ho_acc = accuracy(inp.y[split:], pred_te)
    ho_p, ho_r = precision_recall(
        inp.y[split:], pred_te, average="macro", labels=labels, zero_division=0.0
    )

    return ValidationAgentOutput(
        cv_accuracy_mean=acc_m,
        cv_precision_macro_mean=p_m,
        cv_recall_macro_mean=r_m,
        n_folds=len(fold_rows),
        fold_details=tuple(fold_rows),
        leakage_risk_messages=leakage_msgs,
        holdout_oos_accuracy=float(ho_acc),
        holdout_oos_precision_macro=float(ho_p),
        holdout_oos_recall_macro=float(ho_r),
    )
