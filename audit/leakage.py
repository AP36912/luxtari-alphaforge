"""
Heuristic **obvious** leakage screens (not a formal proof of causal validity).

Flags risky **feature names** and extreme correlation with **next-bar** returns
(momentum features may still correlate; tune thresholds via :class:`AuditConfig`).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from audit.results import AuditReport

# Substrings that strongly suggest future-peeking or target leakage in column names
_NAME_FORBIDDEN = re.compile(
    r"(^|_)(future|fwd|forward|lead|shift_neg|tplus|t_plus|next_bar|"
    r"target|label|y_true|y_pred)(_|$)",
    re.IGNORECASE,
)


def audit_feature_names(feature_names: Sequence[str]) -> AuditReport:
    rep = AuditReport()
    bad = [n for n in feature_names if _NAME_FORBIDDEN.search(str(n))]
    if bad:
        rep.add(
            "error",
            "LEAK_NAME",
            f"Feature names suggest lookahead/target leakage: {bad[:20]}"
            + (" ..." if len(bad) > 20 else ""),
        )
    else:
        rep.add("info", "LEAK_NAME", "No forbidden name patterns in feature list")
    return rep


def audit_correlation_vs_next_return(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    price_col: str = "close",
    max_abs_corr: float = 0.995,
) -> AuditReport:
    """
    Pearson correlation of each feature with **next-bar** simple return.

    Very high |corr| can indicate accidental inclusion of forward prices; threshold is
    intentionally high to reduce false positives on momentum-style features.
    """
    rep = AuditReport()
    if price_col not in df.columns:
        rep.add("warning", "LEAK_CORR", f"Missing {price_col!r}; skip return correlation screen")
        return rep

    close = pd.to_numeric(df[price_col], errors="coerce")
    nret = (close.shift(-1) / close - 1.0).astype(np.float64)
    if nret.notna().sum() < 10:
        rep.add("warning", "LEAK_CORR", "Too few rows for correlation screen")
        return rep

    suspicious: list[tuple[str, float]] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        m = x.notna() & nret.notna()
        if int(m.sum()) < 10:
            continue
        corr = float(np.corrcoef(x[m].to_numpy(), nret[m].to_numpy())[0, 1])
        if not np.isfinite(corr):
            continue
        if abs(corr) >= max_abs_corr:
            suspicious.append((c, corr))

    if suspicious:
        top = sorted(suspicious, key=lambda t: abs(t[1]), reverse=True)[:15]
        rep.add(
            "warning",
            "LEAK_CORR",
            f"Features with |corr(feature, next_return)| >= {max_abs_corr}: "
            + ", ".join(f"{n}={v:.4f}" for n, v in top),
        )
    else:
        rep.add(
            "info",
            "LEAK_CORR",
            f"No features exceed |corr|>={max_abs_corr} vs next-bar return",
        )
    return rep
