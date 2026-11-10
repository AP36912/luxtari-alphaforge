"""Helpers to verify live persistence / restart integrity (Phase 16.5)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from live.paper_broker import PaperBroker
from live.scheduler import LiveScheduler
from live.state_store import LivePersistedState

logger = logging.getLogger("quant_ai.live.restart")


def verify_persisted_state(
    state: LivePersistedState,
    *,
    broker: PaperBroker,
) -> list[str]:
    """
    Cross-check broker snapshot vs persisted ``paper`` dict (after :func:`paper_broker_apply_dict`).
    """
    errors: list[str] = []
    cash_snap = float(state.paper.get("cash", 0.0))
    if abs(cash_snap - broker.state.cash) > 1e-6:
        errors.append(f"cash_mismatch persisted={cash_snap} broker={broker.state.cash}")
    pos_p = (state.paper.get("positions") or [])[:1]
    pos_b = broker.open_position()
    if bool(pos_p) != (pos_b is not None):
        errors.append("open_position_presence_mismatch")
    if pos_b is not None and pos_p:
        e = float(pos_p[0].get("entry_price", 0.0))
        if abs(e - pos_b.entry_price) > 1e-9:
            errors.append("entry_price_mismatch")
    if errors:
        logger.error("restart_verify_failed errors=%s", errors)
    else:
        logger.info("restart_verify_ok paper_state_consistent=True")
    return errors


def verify_last_processed_timestamp(
    state: LivePersistedState,
    buffer: pd.DataFrame,
) -> list[str]:
    """Ensure last buffer bar is not before persisted cursor (monotonic progress)."""
    errors: list[str] = []
    if state.last_processed_timestamp is None or buffer.empty:
        return errors
    last_buf = pd.Timestamp(buffer["timestamp"].iloc[-1])
    last_p = pd.Timestamp(state.last_processed_timestamp)
    if last_buf < last_p:
        errors.append("buffer_ends_before_last_processed_timestamp")
    return errors


def verify_scheduler_after_restore(sched: LiveScheduler, *, expected_last_ts: str | None) -> list[str]:
    """Confirm scheduler replay guard matches expected last processed bar."""
    errors: list[str] = []
    if expected_last_ts is not None and sched._last_processed_bar_ts != expected_last_ts:  # noqa: SLF001
        errors.append(
            f"last_processed_bar_ts_mismatch expected={expected_last_ts!r} got={sched._last_processed_bar_ts!r}"
        )
    if errors:
        logger.error("scheduler_restore_verify_failed errors=%s", errors)
    else:
        logger.info("scheduler_restore_verify_ok")
    return errors
