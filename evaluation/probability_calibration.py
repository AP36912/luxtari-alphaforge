"""
Leakage-safe probability calibration helpers for walk-forward (train-only holdout).

The base classifier is fit on an early time-ordered slice of the training window;
calibration is fit on raw ``predict_proba`` scores on a disjoint tail slice only.
The test window is never used.

sklearn 1.8's ``CalibratedClassifierCV`` with ``FrozenEstimator`` still runs inner
cross-validation and re-fits; we instead call ``_fit_calibrator`` on the base
model's probabilities for the calibration rows only (same mapping as sklearn's
multiclass calibration).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.calibration import _fit_calibrator
from sklearn.preprocessing import label_binarize
from sklearn.utils._response import _get_response_values

CalibrationMethod = Literal["sigmoid", "isotonic"]


def time_ordered_fit_cal_split(
    train_idx: np.ndarray,
    y_all: np.ndarray,
    *,
    holdout_fraction: float,
    min_cal_rows: int,
    min_fit_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sort ``train_idx`` by row order (time). Return ``(fit_idx, cal_idx)`` where
    ``cal_idx`` is the last contiguous slice of the training window.

    Returns None if constraints cannot be satisfied.
    """
    tr = np.sort(np.asarray(train_idx, dtype=np.int64))
    n = int(len(tr))
    if n < min_fit_rows + min_cal_rows:
        return None
    m = max(min_cal_rows, int(round(float(holdout_fraction) * n)))
    m = min(m, n - min_fit_rows)
    if m < min_cal_rows or n - m < min_fit_rows:
        return None
    fit_i = tr[:-m]
    cal_i = tr[-m:]
    y_fit = y_all[fit_i]
    y_cal = y_all[cal_i]
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
        return None
    # Base model is fit only on ``y_fit``; calibration labels must not include
    # a class that never appeared there (otherwise ``predict_proba`` has no column).
    fit_labels = set(np.unique(y_fit).tolist())
    if not set(np.unique(y_cal).tolist()).issubset(fit_labels):
        return None
    return fit_i, cal_i


def multiclass_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    classes: np.ndarray,
) -> float:
    """Mean squared error between one-hot ``y`` and ``proba`` (same ``classes`` order as proba columns)."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    cls = np.asarray(classes, dtype=np.int64)
    # Align columns: sklearn proba columns follow base.classes_
    Y = label_binarize(y, classes=cls)
    if Y.shape[1] == 1 and len(cls) == 2:
        Y = np.hstack([1.0 - Y, Y])
    if p.shape != Y.shape:
        raise ValueError(f"shape mismatch proba {p.shape} vs labels {Y.shape}")
    return float(np.mean(np.sum((p - Y) ** 2, axis=1)))


def top_class_confidence_ece(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error for the predicted top class: bin by max proba,
    compare mean confidence to accuracy of that top-1 decision.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    cls = np.asarray(classes, dtype=np.int64)
    pred_label = cls[np.argmax(p, axis=1)]
    conf = p.max(axis=1)
    correct = (pred_label == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        ece += w * abs(float(np.mean(correct[mask]) - float(np.mean(conf[mask]))))
    return float(ece)


class LeakageSafeCalibratedClassifier:
    """Thin wrapper so walk-forward can call ``predict`` / ``predict_proba`` like sklearn."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.classes_ = np.asarray(inner.classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._inner.predict_proba(X), dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.classes_[idx]


def fit_prefit_calibrator(
    base_fitted: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    method: CalibrationMethod,
) -> LeakageSafeCalibratedClassifier:
    """
    Calibrate an **already-fitted** sklearn classifier using only ``(X_cal, y_cal)``.

    Uses the base model's ``predict_proba`` (or decision_function) on the
    calibration slice as inputs to sklearn's per-class isotonic / sigmoid
    calibrators — no inner cross-validation and no refit of the forest on the
    calibration rows.

    Raises ``ValueError`` if ``y_cal`` contains a class label not present in
    ``base_fitted.classes_`` (the fit slice did not expose that class, so
    probabilities for it are undefined).
    """
    Xc = np.asarray(X_cal, dtype=np.float64)
    yc = np.asarray(y_cal, dtype=np.int64)
    allowed = np.asarray(base_fitted.classes_)
    if not np.isin(yc, allowed).all():
        unseen = np.unique(yc[~np.isin(yc, allowed)])
        raise ValueError(
            f"calibration y contains class(es) {unseen.tolist()} absent from base.classes_"
        )

    predictions, _ = _get_response_values(
        base_fitted,
        Xc,
        response_method=["decision_function", "predict_proba"],
    )
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    classes = np.asarray(base_fitted.classes_)
    # sklearn 1.5.x private API: _fit_calibrator does not accept xp=
    try:
        inner = _fit_calibrator(
            base_fitted,
            predictions,
            yc,
            classes,
            str(method),
            sample_weight=None,
        )
    except TypeError:
        inner = _fit_calibrator(
            base_fitted,
            predictions,
            yc,
            classes,
            str(method),
        )
    return LeakageSafeCalibratedClassifier(inner)


def calibration_holdout_metrics(
    base_fitted: Any,
    cal_fitted: LeakageSafeCalibratedClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Brier and ECE on the calibration holdout (train tail only)."""
    Xc = np.asarray(X_cal, dtype=np.float64)
    yc = np.asarray(y_cal, dtype=np.int64)
    cls = np.asarray(classes, dtype=np.int64)
    p_uncal = np.asarray(base_fitted.predict_proba(Xc), dtype=np.float64)
    p_cal = np.asarray(cal_fitted.predict_proba(Xc), dtype=np.float64)
    return {
        "brier_multiclass_uncalibrated": round(multiclass_brier_score(yc, p_uncal, classes=cls), 8),
        "brier_multiclass_calibrated": round(multiclass_brier_score(yc, p_cal, classes=cls), 8),
        "top_class_ece_uncalibrated": round(
            top_class_confidence_ece(yc, p_uncal, classes=cls), 8
        ),
        "top_class_ece_calibrated": round(top_class_confidence_ece(yc, p_cal, classes=cls), 8),
    }


__all__ = [
    "CalibrationMethod",
    "LeakageSafeCalibratedClassifier",
    "calibration_holdout_metrics",
    "fit_prefit_calibrator",
    "multiclass_brier_score",
    "time_ordered_fit_cal_split",
    "top_class_confidence_ece",
]
