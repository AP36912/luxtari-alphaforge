"""Framework audit: timestamps, leakage heuristics, splits, backtest alignment, repro."""

from audit.backtest_alignment import audit_backtest_inputs
from audit.leakage import audit_correlation_vs_next_return, audit_feature_names
from audit.repro import assert_numpy_determinism_after_seed, environment_snapshot, fingerprint_json_blob
from audit.results import AuditFinding, AuditReport, audit_findings_as_jsonable, merge_audit_reports
from audit.splits import (
    audit_holdout_prefix_split,
    audit_purged_kfold_fold,
    describe_purged_kfold_temporal_overlap,
)
from audit.timestamps import assert_row_alignment, audit_timestamps

__all__ = [
    "AuditFinding",
    "AuditReport",
    "audit_findings_as_jsonable",
    "merge_audit_reports",
    "assert_numpy_determinism_after_seed",
    "assert_row_alignment",
    "audit_backtest_inputs",
    "audit_correlation_vs_next_return",
    "audit_feature_names",
    "audit_holdout_prefix_split",
    "audit_purged_kfold_fold",
    "audit_timestamps",
    "describe_purged_kfold_temporal_overlap",
    "environment_snapshot",
    "fingerprint_json_blob",
]
