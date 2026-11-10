"""Timestamp monotonicity and duplicate checks (alignment with bar order)."""

from __future__ import annotations

import pandas as pd

from audit.results import AuditReport


def audit_timestamps(df: pd.DataFrame, *, col: str = "timestamp") -> AuditReport:
    """
    Verify ``timestamp`` is non-decreasing and has no duplicates (common PIT requirement).

    Missing column → error.
    """
    rep = AuditReport()
    if col not in df.columns:
        rep.add("error", "TS_MISSING", f"Column {col!r} not found")
        return rep

    ts = df[col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        rep.add("warning", "TS_DTYPE", f"{col} is not datetime64; cannot fully verify ordering")

    dup = ts.duplicated()
    if dup.any():
        rep.add("error", "TS_DUPLICATE", f"Found {int(dup.sum())} duplicate {col} values")

    if len(df) >= 2:
        if ts.is_monotonic_increasing:
            rep.add("info", "TS_MONOTONIC", f"{col} is strictly/effectively sorted (non-decreasing)")
        else:
            rep.add("error", "TS_ORDER", f"{col} is not monotonic non-decreasing — sort data before modeling")

    return rep


def assert_row_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
) -> AuditReport:
    """Two frames with the same row count should have identical timestamps row-wise."""
    rep = AuditReport()
    if len(left) != len(right):
        rep.add(
            "error",
            "ALIGN_LEN",
            f"Row count mismatch: left={len(left)} right={len(right)}",
        )
        return rep
    if ts_col in left.columns and ts_col in right.columns:
        l = pd.to_datetime(left[ts_col], errors="coerce")
        r = pd.to_datetime(right[ts_col], errors="coerce")
        if not (l.values == r.values).all():
            rep.add("error", "ALIGN_TS", f"{ts_col} values differ between aligned frames")
        else:
            rep.add("info", "ALIGN_TS", f"{ts_col} row alignment verified ({len(left)} rows)")
    else:
        rep.add("info", "ALIGN_LEN", f"Same row count ({len(left)}); timestamp column not in both frames")
    return rep
