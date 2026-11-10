"""
Modeling agent: fit registry models on a fixed chronological split and compare holdout metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.types import ModelComparisonRow, ModelingAgentOutput
from agents.utils import seed_model_params
from models.config import ModelConfig
from models.registry import create_model
from models.training import ModelTrainer
from validation.metrics import accuracy, precision_recall, roc_auc


@dataclass
class ModelingAgentInput:
    X: np.ndarray
    y: np.ndarray
    split: int
    """First row index of holdout (train is ``[0, split)``)."""
    model_configs: tuple[ModelConfig, ...]
    base_seed: int
    labels_for_metrics: list[int]


def run_modeling_agent(inp: ModelingAgentInput) -> ModelingAgentOutput:
    """
    Train each configured model on the prefix, evaluate on the suffix; pick a deterministic winner.

    Winner = highest holdout accuracy; ties broken lexicographically by ``model_type`` string
    (registry name), then by ``slot_index``.
    """
    if not inp.model_configs:
        raise ValueError("model_configs must be non-empty")
    split = inp.split
    X_tr, X_te = inp.X[:split], inp.X[split:]
    y_tr, y_te = inp.y[:split], inp.y[split:]

    rows: list[ModelComparisonRow] = []
    for slot, mc in enumerate(inp.model_configs):
        extra = seed_model_params(dict(mc.extra_params), inp.base_seed + 1000 + slot)
        mconf = ModelConfig(name=mc.name, extra_params=extra)
        model = create_model(mconf)
        trainer = ModelTrainer(model)
        trainer.fit(X_tr, y_tr)
        pred_te = trainer.predict(X_te)
        acc = accuracy(y_te, pred_te)
        p_mac, r_mac = precision_recall(
            y_te, pred_te, average="macro", labels=inp.labels_for_metrics, zero_division=0.0
        )
        roc = None
        if model.supports_proba:
            try:
                proba = trainer.predict_proba(X_te)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    roc = roc_auc(y_te, proba, labels=inp.labels_for_metrics)
            except Exception:  # noqa: BLE001
                roc = None

        rows.append(
            ModelComparisonRow(
                model_name=mc.name,
                model_type=mc.name,
                holdout_accuracy=float(acc),
                holdout_precision_macro=float(p_mac),
                holdout_recall_macro=float(r_mac),
                roc_auc_ovo=float(roc) if roc is not None and np.isfinite(roc) else None,
            )
        )

    best_idx = min(
        range(len(rows)),
        key=lambda i: (-rows[i].holdout_accuracy, rows[i].model_name, rows[i].model_type, i),
    )
    chosen = inp.model_configs[best_idx]

    return ModelingAgentOutput(
        comparisons=tuple(rows),
        chosen_model_name=chosen.name,
        chosen_model_config=chosen,
    )
