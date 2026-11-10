"""Train-only probability calibration split + metrics."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from evaluation.probability_calibration import (
    calibration_holdout_metrics,
    fit_prefit_calibrator,
    multiclass_brier_score,
    time_ordered_fit_cal_split,
    top_class_confidence_ece,
)


def test_time_ordered_split_respects_constraints() -> None:
    tr = np.arange(200, dtype=np.int64)
    y = np.tile(np.array([-1, 0, 1], dtype=np.int64), 67)[:200]
    out = time_ordered_fit_cal_split(
        tr, y, holdout_fraction=0.15, min_cal_rows=20, min_fit_rows=80
    )
    assert out is not None
    fit_i, cal_i = out
    assert len(np.intersect1d(fit_i, cal_i)) == 0
    assert cal_i.min() > fit_i.max()


def test_calibrated_brier_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 8
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    y = np.where(y == 0, -1, 1)
    y[::7] = 0
    fit_i = np.arange(300)
    cal_i = np.arange(300, 400)
    clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
    clf.fit(X[fit_i], y[fit_i])
    cal = fit_prefit_calibrator(clf, X[cal_i], y[cal_i], method="sigmoid")
    cls = np.asarray(cal.classes_)
    met = calibration_holdout_metrics(clf, cal, X[cal_i], y[cal_i], classes=cls)
    assert 0.0 <= met["brier_multiclass_calibrated"] <= 2.0
    assert met["brier_multiclass_uncalibrated"] >= 0.0


def test_top_class_ece_uses_class_order() -> None:
    classes = np.array([-1, 0, 1])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([-1, 0])
    e = top_class_confidence_ece(y, p, classes=classes)
    assert 0.0 <= e <= 1.0


def test_multiclass_brier_shape() -> None:
    classes = np.array([-1, 0, 1])
    y = np.array([-1, 0, 1, -1])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.9, 0.05, 0.05]])
    b = multiclass_brier_score(y, p, classes=classes)
    assert b >= 0.0
