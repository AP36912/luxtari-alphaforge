"""Tests for heuristic leakage screens (names, correlation vs next return)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit.leakage import audit_correlation_vs_next_return, audit_feature_names


def test_audit_feature_names_flags_suspicious_substrings() -> None:
    rep = audit_feature_names(["rsi_14", "my_forward_return"])
    assert any(f.severity == "error" and f.code == "LEAK_NAME" for f in rep.findings)


def test_audit_feature_names_clean_list_info_only() -> None:
    rep = audit_feature_names(["rsi_14", "atr_20"])
    assert not rep.errors()
    assert any(f.code == "LEAK_NAME" and f.severity == "info" for f in rep.findings)


def test_audit_correlation_warns_when_feature_equals_next_return() -> None:
    n = 60
    close = pd.Series(np.linspace(100.0, 160.0, n), dtype=np.float64)
    nret = close.shift(-1) / close - 1.0
    df = pd.DataFrame({"close": close, "leaky": nret})
    rep = audit_correlation_vs_next_return(df, ["leaky"], max_abs_corr=0.99)
    warns = [f for f in rep.findings if f.code == "LEAK_CORR" and f.severity == "warning"]
    assert warns, "expected LEAK_CORR warning for feature identical to next return"
