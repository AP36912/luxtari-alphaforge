from __future__ import annotations

"""Offline sim broker adapter for the public candidate (no live submit).
Sanitized export replacement — not a byte copy of the private original."""

import hashlib
from typing import Any

from broker.base import BrokerAdapter
from broker.models import (
    AccountState,
    CancelRequest,
    OrderRequest,
    OrderResult,
    PositionCloseRequest,
    PositionSnapshot,
    utc_now_iso,
)


class DemoBrokerAdapter(BrokerAdapter):
    """
    Stub useful for gateway tests: ``submit_order`` succeeds unless synthetic hash ends in ``0``.
    Does not mutate :class:`~live.paper_broker.PaperBroker`.
    """

    name = "broker_demo"

    def __init__(self, *, reject_if_hash_suffix: str = "0") -> None:
        self._connected = False
        self._reject_suffix = reject_if_hash_suffix

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def health_check(self) -> tuple[bool, str]:
        return self._connected, "demo_connected" if self._connected else "demo_not_connected"

    def get_account_state(self) -> AccountState:
        return AccountState(equity=100_000.0, cash=100_000.0, currency="DEMO")

    def get_open_positions(self) -> list[PositionSnapshot]:
        return []

    def get_open_orders(self) -> list[dict[str, Any]]:
        return []

    def submit_order(self, req: OrderRequest) -> OrderResult:
        h = hashlib.sha256(
            f"{req.symbol}:{req.target_direction}:{req.timestamp}".encode()
        ).hexdigest()
        ok = not h.endswith(self._reject_suffix)
        return OrderResult(
            timestamp_utc=utc_now_iso(),
            success=ok,
            broker_order_id=f"DEMO-{h[:12]}" if ok else None,
            message="demo_accept" if ok else "demo_reject_synthetic",
            rejection_reason=None if ok else "demo_synthetic_reject",
            paper_fills_count=0,
        )

    def cancel_order(self, req: CancelRequest) -> OrderResult:
        return OrderResult(
            timestamp_utc=utc_now_iso(),
            success=False,
            broker_order_id=req.broker_order_id,
            message="demo_cancel_not_supported",
            rejection_reason="not_supported",
        )

    def close_position(self, req: PositionCloseRequest) -> OrderResult:
        return OrderResult(
            timestamp_utc=utc_now_iso(),
            success=False,
            message="demo_close_not_supported",
            rejection_reason="not_supported",
        )
