"""
Chronological split checks for time-series experiments.

Purged K-fold **intentionally** mixes train indices before/after a test block; only
use strict temporal checks where appropriate (e.g. prefix holdout).
"""

from __future__ import annotations

import numpy as np

from audit.results import AuditReport


def audit_holdout_prefix_split(split: int, n_samples: int) -> AuditReport:
    """
    Holdout must be ``[0, split)`` train and ``[split, n)`` test (contiguous, time-ordered).
    """
    rep = AuditReport()
    if split <= 0:
        rep.add("error", "SPLIT_HOLDOUT", f"split must be positive, got {split}")
    if split >= n_samples:
        rep.add("error", "SPLIT_HOLDOUT", f"split ({split}) must be < n_samples ({n_samples})")
    if not rep.errors():
        rep.add(
            "info",
            "SPLIT_HOLDOUT",
            f"Holdout uses prefix train [0,{split}) and test [{split},{n_samples})",
        )
    return rep


def audit_purged_kfold_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_samples: int,
) -> AuditReport:
    """
    Sanity checks for one purged fold: non-empty, within bounds, disjoint, test block contiguous.
    """
    rep = AuditReport()
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if tr.size == 0:
        rep.add("warning", "SPLIT_PK_EMPTY_TRAIN", "Empty training fold after purge")
    if te.size == 0:
        rep.add("error", "SPLIT_PK_EMPTY_TEST", "Empty test fold")
    if np.max(tr, initial=-1) >= n_samples or np.max(te, initial=-1) >= n_samples:
        rep.add("error", "SPLIT_PK_BOUNDS", "Fold indices out of bounds")
    if np.intersect1d(tr, te).size > 0:
        rep.add("error", "SPLIT_PK_OVERLAP", "Train and test indices overlap")

    if te.size >= 2:
        te_sorted = np.sort(te)
        contiguous = bool(np.all(te_sorted[1:] == te_sorted[:-1] + 1))
        if not contiguous:
            rep.add("warning", "SPLIT_PK_TEST_SHAPE", "Test indices are not a contiguous block")
        else:
            rep.add("info", "SPLIT_PK_TEST", f"Test fold is contiguous ({te.size} bars)")
    return rep


def describe_purged_kfold_temporal_overlap(train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, int | bool]:
    """Diagnostics: whether any train index falls inside the test index span (allowed after purge)."""
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if te.size == 0:
        return {"train_in_test_span": 0, "max_train": -1, "min_test": -1, "max_test": -1}
    t0, t1 = int(np.min(te)), int(np.max(te))
    in_span = int(np.sum((tr >= t0) & (tr <= t1))) if tr.size else 0
    return {
        "train_in_test_span": in_span,
        "max_train": int(np.max(tr)) if tr.size else -1,
        "min_test": t0,
        "max_test": t1,
    }
