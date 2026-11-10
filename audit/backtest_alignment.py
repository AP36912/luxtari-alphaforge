"""Verify predictions / signals align with OHLC rows for :class:`backtest.engine.BacktestEngine`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit.results import AuditReport


def audit_backtest_inputs(
    ohlc: pd.DataFrame,
    signals: np.ndarray,
    *,
    predictions_source: str = "holdout_model_predictions",
) -> AuditReport:
    """
    Same-length OHLC and signals; finite signals in [-1, 1] (engine contract).

    The engine executes ``signals[t]`` at ``open[t+1]`` — no extra row required here;
    last-bar signal is intentionally not traded (engine contract).
    """
    rep = AuditReport()
    n = len(ohlc)
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    if sig.shape[0] != n:
        rep.add(
            "error",
            "BT_ALIGN_LEN",
            f"OHLC rows ({n}) != signals length ({sig.shape[0]}); source={predictions_source}",
        )
        return rep

    for col in ("open", "close"):
        if col not in ohlc.columns:
            rep.add("error", "BT_OHLC", f"Missing required column {col!r}")

    if not np.all(np.isfinite(sig)):
        rep.add("error", "BT_SIGNAL_FINITE", "Non-finite values in signals array")

    if np.max(np.abs(sig)) > 1.000001:
        rep.add(
            "warning",
            "BT_SIGNAL_RANGE",
            "Signals should lie in [-1, 1] for standard sizing contract",
        )

    rep.add(
        "info",
        "BT_ALIGN",
        f"Backtest alignment OK: {n} bars, signals from {predictions_source}",
    )
    return rep
