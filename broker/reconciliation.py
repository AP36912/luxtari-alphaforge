"""
Reconcile local tracking with broker-reported open orders and positions (Phase 18).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from broker.base import BrokerAdapter
from broker.order_tracker import OrderTracker


@dataclass
class ReconciliationReport:
    """Summary of broker vs local tracker state."""

    checked_at_utc: str
    broker_positions_count: int
    broker_orders_count: int
    tracked_clients: int
    tracked_brokers: int
    orphan_broker_orders: list[str] = field(default_factory=list)
    """Broker order ids with no local client mapping (informational)."""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at_utc": self.checked_at_utc,
            "broker_positions_count": self.broker_positions_count,
            "broker_orders_count": self.broker_orders_count,
            "tracked_clients": self.tracked_clients,
            "tracked_brokers": self.tracked_brokers,
            "orphan_broker_orders": self.orphan_broker_orders,
            "notes": self.notes,
        }


def reconcile_broker_vs_tracker(
    adapter: BrokerAdapter,
    tracker: OrderTracker,
    *,
    connect_if_needed: bool = True,
) -> ReconciliationReport:
    """
    Pull open orders/positions from ``adapter`` and compare ticket coverage vs tracker.

    Does not mutate broker state. ``connect_if_needed`` calls ``connect()`` when False health.
    """
    now = datetime.now(timezone.utc).isoformat()
    notes: list[str] = []
    if connect_if_needed:
        ok, msg = adapter.health_check()
        if not ok:
            adapter.connect()
            ok2, msg2 = adapter.health_check()
            notes.append(f"reconnect_attempt ok={ok2} msg={msg2}")
            if not ok2:
                notes.append(f"prior_health msg={msg}")

    positions = adapter.get_open_positions()
    orders = adapter.get_open_orders()
    broker_tickets = {str(o.get("ticket", "")) for o in orders if o.get("ticket") is not None}

    tracked_broker_keys = set()
    # OrderTracker stores by_broker — we don't expose keys; use snapshot via getattr
    by_b = getattr(tracker, "_by_broker", {})
    tracked_broker_keys = set(by_b.keys())

    orphans = sorted(broker_tickets - tracked_broker_keys) if broker_tickets else []

    return ReconciliationReport(
        checked_at_utc=now,
        broker_positions_count=len(positions),
        broker_orders_count=len(orders),
        tracked_clients=len(getattr(tracker, "_by_client", {})),
        tracked_brokers=len(tracked_broker_keys),
        orphan_broker_orders=orphans[:50],
        notes=notes,
    )


def detect_lifecycle_mismatches(
    *,
    phase: Literal["post_open", "post_close"],
    submit_succeeded: bool | None = None,
    close_succeeded: bool | None = None,
    broker_order_id: str | None = None,
    deal_id: str | None = None,
    symbol: str,
    position_ticket: str | None,
    positions: list[Any],
    open_orders: list[dict[str, Any]],
    orphan_position_symbols: list[str] | None = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Return ``(mismatches, warnings)`` for demo lifecycle verification (Phase 19.5).

    ``positions`` should be :class:`~broker.models.PositionSnapshot` or compatible
    (``symbol``, ``broker_position_id``).
    """
    mismatches: list[str] = []
    warnings: list[str] = []
    sym_u = str(symbol or "").upper()

    def _pos_match() -> list[Any]:
        return [p for p in positions if str(getattr(p, "symbol", "")).upper() == sym_u]

    if phase == "post_open":
        if submit_succeeded is True:
            if not broker_order_id and not deal_id:
                code = "submit_success_missing_broker_identifiers"
                (mismatches if strict else warnings).append(code)
            matching = _pos_match()
            if not matching:
                msg = "submit_success_no_visible_position_for_symbol"
                (mismatches if strict else warnings).append(msg)
        if orphan_position_symbols:
            for s in orphan_position_symbols:
                if str(s).upper() != sym_u:
                    warnings.append(f"orphan_position_other_symbol:{s}")

    if phase == "post_close":
        if close_succeeded is True:
            matching = _pos_match()
            if position_ticket:
                still = [
                    p
                    for p in positions
                    if str(getattr(p, "broker_position_id", "") or "") == str(position_ticket)
                ]
                if still:
                    mismatches.append("close_success_position_ticket_still_open")
            elif matching:
                mismatches.append("close_success_symbol_position_still_open")
        if len(open_orders) > 0 and strict:
            mismatches.append(f"post_close_open_orders_remain_count={len(open_orders)}")
        elif len(open_orders) > 0:
            warnings.append(f"post_close_open_orders_remain_count={len(open_orders)}")

    return mismatches, warnings


def reconcile_broker_vs_gateway_context(
    *,
    adapter: BrokerAdapter,
    tracker: OrderTracker,
    gateway_broker_order_id: str | None,
    gateway_deal_id: str | None,
    connect_if_needed: bool = True,
) -> tuple[ReconciliationReport, list[str]]:
    """
    Run :func:`reconcile_broker_vs_tracker` and add notes if gateway IDs are absent from tracker.
    """
    rep = reconcile_broker_vs_tracker(adapter, tracker, connect_if_needed=connect_if_needed)
    extra: list[str] = []
    by_b = getattr(tracker, "_by_broker", {})
    if gateway_broker_order_id and str(gateway_broker_order_id) not in by_b:
        extra.append(f"gateway_broker_order_id_not_in_tracker:{gateway_broker_order_id}")
    if gateway_deal_id and str(gateway_deal_id) not in by_b:
        extra.append("gateway_deal_id_not_tracked_separately")
    rep.notes = list(rep.notes) + extra
    return rep, extra
