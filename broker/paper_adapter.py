"""
Paper execution via existing :class:`live.paper_broker.PaperBroker` (Phase 17).
"""

from __future__ import annotations

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
from live.paper_broker import PaperBroker, PaperFill


class PaperBrokerAdapter(BrokerAdapter):
    name = "paper"

    def __init__(self, broker: PaperBroker) -> None:
        self._b = broker

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def health_check(self) -> tuple[bool, str]:
        return True, "paper_broker_ok"

    def get_account_state(self, last_mark: float | None = None) -> AccountState:
        """Return marked equity when a current or previously observed close is available."""
        cash = float(self._b.state.cash)
        mark = self._b.last_close if last_mark is None else float(last_mark)
        equity = cash if mark is None else float(self._b.equity(mark))
        return AccountState(
            cash=cash,
            equity=equity,
            raw={"realized_pnl": self._b.state.realized_pnl, "last_mark": mark},
        )

    def get_open_positions(self) -> list[PositionSnapshot]:
        pos = self._b.open_position()
        if pos is None:
            return []
        return [
            PositionSnapshot(
                symbol="",
                side=pos.side,
                quantity=pos.quantity,
                avg_price=pos.entry_price,
            )
        ]

    def get_open_orders(self) -> list[dict[str, Any]]:
        return []

    def execute_bar_intent(self, req: OrderRequest) -> tuple[OrderResult, list[PaperFill]]:
        fills = self._b.step(
            req.timestamp,
            req.bar_open,
            req.bar_high,
            req.bar_low,
            req.bar_close,
            req.target_direction,
        )
        return (
            OrderResult(
                timestamp_utc=utc_now_iso(),
                success=True,
                broker_order_id=None,
                message="paper_fill",
                paper_fills_count=len(fills),
                extra={"fills": [_fill_dict(f) for f in fills]},
            ),
            fills,
        )

    def submit_order(self, req: OrderRequest) -> OrderResult:
        res, _ = self.execute_bar_intent(req)
        return res

    def cancel_order(self, req: CancelRequest) -> OrderResult:
        return OrderResult(
            timestamp_utc=utc_now_iso(),
            success=False,
            message="paper_adapter_cancel_not_supported",
            rejection_reason="not_supported",
        )

    def close_position(self, req: PositionCloseRequest) -> OrderResult:
        # Flatten via step(..., 0) at last known close — caller should pass bar in OrderRequest path
        return OrderResult(
            timestamp_utc=utc_now_iso(),
            success=False,
            message="use_submit_order_with_target_direction_zero",
            rejection_reason="use_gateway_bar_path",
        )


def _fill_dict(f: PaperFill) -> dict[str, Any]:
    return {
        "timestamp": str(f.timestamp),
        "side": f.side,
        "quantity": f.quantity,
        "price": f.price,
        "commission": f.commission,
        "reason": f.reason,
    }
