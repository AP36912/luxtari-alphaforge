"""
Internal ↔ broker order id mapping and submit deduplication (Phase 18).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from broker.models import OrderRequest, OrderResult


def order_submit_fingerprint(req: OrderRequest) -> str:
    """Stable fingerprint for idempotency (same bar intent should not double-submit)."""
    ts = str(req.timestamp)
    payload = f"{req.symbol}|{ts}|{req.target_direction}|{req.side}|{req.quantity:.8f}|{req.order_type}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass
class TrackedOrder:
    client_order_id: str | None
    broker_order_id: str | None
    fingerprint: str
    status: str
    created_mono: float
    extra: dict[str, Any] = field(default_factory=dict)


class OrderTracker:
    """
    Maps client ids to broker tickets and blocks duplicate fingerprints after a **successful** submit.
    """

    def __init__(self, *, fingerprint_ttl_seconds: float = 120.0) -> None:
        self._by_client: dict[str, TrackedOrder] = {}
        self._by_broker: dict[str, TrackedOrder] = {}
        self._success_fp_time: dict[str, float] = {}
        self._ttl = fingerprint_ttl_seconds

    def _prune_success(self, now_mono: float) -> None:
        cutoff = now_mono - self._ttl
        dead = [k for k, t in self._success_fp_time.items() if t < cutoff]
        for k in dead:
            del self._success_fp_time[k]

    def should_block_duplicate_after_success(
        self, req: OrderRequest, *, now_mono: float, ttl_seconds: float
    ) -> bool:
        """If TTL > 0 and same intent already succeeded recently, block (no duplicate orders)."""
        if ttl_seconds <= 0:
            return False
        self._prune_success(now_mono)
        fp = order_submit_fingerprint(req)
        t = self._success_fp_time.get(fp)
        return t is not None and (now_mono - t) < float(ttl_seconds)

    def mark_fingerprint_success(self, req: OrderRequest, *, now_mono: float) -> None:
        fp = order_submit_fingerprint(req)
        self._success_fp_time[fp] = now_mono

    def register_result(self, req: OrderRequest, res: OrderResult, *, now_mono: float) -> None:
        fp = order_submit_fingerprint(req)
        cid = req.client_order_id or ""
        to = TrackedOrder(
            client_order_id=req.client_order_id,
            broker_order_id=res.broker_order_id,
            fingerprint=fp,
            status="accepted" if res.success else "rejected",
            created_mono=now_mono,
            extra=dict(res.extra or {}),
        )
        if cid:
            self._by_client[cid] = to
        if res.broker_order_id:
            self._by_broker[str(res.broker_order_id)] = to

    def get_by_client_id(self, client_order_id: str) -> TrackedOrder | None:
        return self._by_client.get(client_order_id)

    def get_by_broker_id(self, broker_order_id: str) -> TrackedOrder | None:
        return self._by_broker.get(broker_order_id)

    def snapshot_json(self) -> str:
        return json.dumps(
            {
                "by_client": {k: v.__dict__ for k, v in self._by_client.items()},
                "by_broker": {k: v.__dict__ for k, v in self._by_broker.items()},
            },
            default=str,
        )
