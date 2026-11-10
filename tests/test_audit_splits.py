"""Tests for chronological holdout and purged K-fold sanity checks."""

from __future__ import annotations

import numpy as np

from audit.results import AuditReport, merge_audit_reports
from audit.splits import audit_holdout_prefix_split, audit_purged_kfold_fold


def test_holdout_prefix_split_errors_when_invalid() -> None:
    rep = audit_holdout_prefix_split(0, 100)
    assert rep.errors()
    rep2 = audit_holdout_prefix_split(100, 100)
    assert rep2.errors()


def test_holdout_prefix_split_ok() -> None:
    rep = audit_holdout_prefix_split(80, 100)
    assert not rep.errors()


def test_purged_fold_detects_overlap() -> None:
    tr = np.array([0, 1, 2], dtype=np.int64)
    te = np.array([2, 3], dtype=np.int64)
    rep = audit_purged_kfold_fold(tr, te, n_samples=10)
    assert any(f.code == "SPLIT_PK_OVERLAP" for f in rep.findings)


def test_merge_audit_reports() -> None:
    a = AuditReport()
    a.add("info", "A", "one")
    b = AuditReport()
    b.add("warning", "B", "two")
    m = merge_audit_reports(a, b)
    assert len(m.findings) == 2
