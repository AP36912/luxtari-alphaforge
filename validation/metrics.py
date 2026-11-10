"""
Classification metrics (accuracy, precision/recall, ROC AUC).

Thin wrappers around scikit-learn with stable defaults for research code.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

AverageArg = Literal["micro", "macro", "weighted", "binary", "samples"]


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Fraction of labels equal to predictions."""
    return float(accuracy_score(y_true, y_pred))


def precision_recall(
    y_true: Any,
    y_pred: Any,
    *,
    average: AverageArg = "binary",
    labels: Any | None = None,
    zero_division: float = 0.0,
) -> tuple[float, float]:
    """
    Precision and recall.

    For multiclass or labels such as ``{-1, 0, 1}``, pass ``labels=`` and set
    ``average`` to ``macro``, ``weighted``, or ``micro`` as needed.
    """
    p, r, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        labels=labels,
        zero_division=zero_division,
    )
    return float(p), float(r)


def roc_auc(
    y_true: Any,
    y_score: Any,
    *,
    average: AverageArg = "macro",
    multi_class: str = "ovr",
    labels: Any | None = None,
) -> float:
    """
    Area under ROC curve.

    ``y_score`` should be positive-class probability (binary) or shape
    ``(n_samples, n_classes)`` for multiclass. For uncommon label sets, pass ``labels=``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return float(roc_auc_score(y_true, y_score))
    return float(
        roc_auc_score(
            y_true,
            y_score,
            average=average,
            multi_class=multi_class,
            labels=labels,
        )
    )
