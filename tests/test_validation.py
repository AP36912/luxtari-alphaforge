"""Validation splits (purge, embargo, time order) and classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from validation.event_bounds import event_bounds_from_horizon
from validation.metrics import accuracy, precision_recall, roc_auc
from validation.purged_kfold import PurgedKFold
from validation.splits import purged_kfold_indices
from validation.time_series_split import PurgedTimeSeriesSplit
from validation.walk_forward import WalkForwardSplitter


def _no_event_overlap(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    es: np.ndarray,
    ee: np.ndarray,
) -> bool:
    """True iff no train event interval intersects any test event interval."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return True
    s_tr = es[train_idx][:, np.newaxis]
    e_tr = ee[train_idx][:, np.newaxis]
    s_te = es[test_idx][np.newaxis, :]
    e_te = ee[test_idx][np.newaxis, :]
    overlap = (s_tr <= e_te) & (s_te <= e_tr)
    return not np.any(overlap)


def test_purged_kfold_indices_helper_matches_purged_kfold() -> None:
    n, k, h = 40, 2, 2
    pkf = PurgedKFold(k, label_horizon=h, embargo=0)
    X = np.zeros(n)
    for (a_tr, a_te), (b_tr, b_te) in zip(
        pkf.split(X),
        purged_kfold_indices(n, k, label_horizon=h, embargo=0),
    ):
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)


def test_purged_kfold_contiguous_folds_cover_all_test_indices() -> None:
    n, k, h = 60, 3, 2
    pkf = PurgedKFold(k, label_horizon=h, embargo=0)
    X = np.arange(n)
    folds = list(pkf.split(X))
    assert len(folds) == k
    seen = np.zeros(n, dtype=bool)
    for _tr, te in folds:
        assert np.all(te[1:] == te[:-1] + 1)
        seen[te] = True
    assert np.all(seen)


def test_purged_kfold_no_event_leakage_after_purge() -> None:
    n, k, h = 100, 4, 5
    es, ee = event_bounds_from_horizon(n, h)
    pkf = PurgedKFold(k, label_horizon=h, embargo=0)
    X = np.zeros(n)
    for tr, te in pkf.split(X):
        assert _no_event_overlap(tr, te, es, ee)


def test_purged_kfold_embargo_removes_train_just_after_test_block() -> None:
    n, k, h = 80, 4, 2
    emb = 3
    pkf = PurgedKFold(k, label_horizon=h, embargo=emb)
    X = np.zeros(n)
    for tr, te in pkf.split(X):
        if len(te) == 0:
            continue
        te_max = int(np.max(te))
        bad = (tr > te_max) & (tr <= te_max + emb)
        assert not np.any(bad)


def test_walk_forward_expanding_time_order_embargo_and_no_overlap() -> None:
    n = 120
    embargo = 2
    wf = WalkForwardSplitter(
        test_size=10,
        window="expanding",
        embargo=embargo,
        label_horizon=3,
        initial_train=30,
    )
    X = np.zeros(n)
    es, ee = event_bounds_from_horizon(n, 3)
    for tr, te in wf.split(X):
        assert len(te) == 10
        assert np.max(tr) < np.min(te)
        gap = np.min(te) - np.max(tr) - 1
        assert gap >= embargo
        assert _no_event_overlap(tr, te, es, ee)


def test_walk_forward_rolling_time_order_and_no_overlap() -> None:
    n = 100
    wf = WalkForwardSplitter(
        test_size=8,
        train_size=25,
        window="rolling",
        step=5,
        embargo=1,
        label_horizon=2,
        initial_train=25,
    )
    X = np.zeros(n)
    es, ee = event_bounds_from_horizon(n, 2)
    for tr, te in wf.split(X):
        assert len(tr) <= 25
        assert np.max(tr) < np.min(te)
        assert _no_event_overlap(tr, te, es, ee)


def test_walk_forward_get_n_splits_matches_iterable_length() -> None:
    n = 200
    wf = WalkForwardSplitter(
        test_size=15,
        window="expanding",
        embargo=2,
        label_horizon=1,
        initial_train=40,
    )
    X = np.zeros(n)
    assert wf.get_n_splits(X) == len(list(wf.split(X)))


def test_purged_time_series_split_chronology_and_no_event_leakage() -> None:
    n = 90
    gap = 2
    ptsc = PurgedTimeSeriesSplit(4, embargo=gap, label_horizon=4)
    X = np.zeros(n)
    es, ee = event_bounds_from_horizon(n, 4)
    for tr, te in ptsc.split(X):
        assert np.max(tr) < np.min(te)
        assert np.min(te) - np.max(tr) - 1 >= gap
        assert _no_event_overlap(tr, te, es, ee)


def test_purged_time_series_get_n_splits() -> None:
    ptsc = PurgedTimeSeriesSplit(3, embargo=0, label_horizon=1)
    assert ptsc.get_n_splits() == 3


def test_metrics_accuracy_precision_recall_auc_binary() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    p, r = precision_recall(y_true, y_pred, average="binary")
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0
    assert accuracy(y_true, y_pred) == pytest.approx(0.8)
    scores = np.array([0.1, 0.9, 0.6, 0.2, 0.85])
    auc = roc_auc(y_true, scores)
    assert 0.0 <= auc <= 1.0


def test_metrics_multiclass_auc_and_macro_pr() -> None:
    y_true = np.array([-1, 0, 1, 1, -1])
    y_pred = np.array([-1, 0, 1, 0, 1])
    labels = [-1, 0, 1]
    p, r = precision_recall(y_true, y_pred, average="macro", labels=labels)
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
        ]
    )
    auc = roc_auc(y_true, y_score, labels=labels)
    assert 0.0 <= auc <= 1.0
