"""
Deterministic validation for incoming live OHLCV rows (Phase 16.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

@dataclass
class BarValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v)


def validate_live_bar_row(
    row: pd.Series | dict[str, Any],
    *,
    require_volume: bool = True,
    allow_zero_volume: bool = True,
) -> BarValidationResult:
    """
    Validate a single bar before merge into the live buffer.

    Does not mutate ``row``. Duplicate timestamps are handled at merge time, not here.
    """
    reasons: list[str] = []
    d = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    ts = d.get("timestamp")
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        reasons.append("missing_timestamp")
    else:
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            reasons.append("invalid_timestamp")

    for col in ("open", "high", "low", "close"):
        if col not in d:
            reasons.append(f"missing_{col}")
        elif not _is_finite_number(d[col]):
            reasons.append(f"non_numeric_{col}")

    if require_volume:
        if "volume" not in d:
            reasons.append("missing_volume")
        elif not _is_finite_number(d["volume"]):
            reasons.append("non_numeric_volume")
        elif float(d["volume"]) < 0:
            reasons.append("negative_volume")
        elif not allow_zero_volume and float(d["volume"]) == 0.0:
            reasons.append("zero_volume_disallowed")

    if not reasons:
        o, h, lo, c = (float(d["open"]), float(d["high"]), float(d["low"]), float(d["close"]))
        eps = 1e-9
        if lo > h + eps:
            reasons.append("low_greater_than_high")
        if h + eps < max(o, c):
            reasons.append("invalid_candle_high_vs_body")
        if lo - eps > min(o, c):
            reasons.append("invalid_candle_low_vs_body")

    return BarValidationResult(ok=len(reasons) == 0, reasons=reasons)


def validate_live_frame(
    df: pd.DataFrame,
    *,
    require_volume: bool = True,
) -> list[tuple[int, BarValidationResult]]:
    """Return list of (row_index, result) for rows that fail validation."""
    out: list[tuple[int, BarValidationResult]] = []
    for i in range(len(df)):
        r = validate_live_bar_row(df.iloc[i], require_volume=require_volume)
        if not r.ok:
            out.append((i, r))
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
