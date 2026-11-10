"""
Abstract broker adapter — all external execution goes through implementations (Phase 17).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from broker.models import (
    AccountState,
    CancelRequest,
    OrderRequest,
    OrderResult,
    PositionCloseRequest,
    PositionSnapshot,
)
import logging

logger = logging.getLogger("quant_ai.broker.base")


class BrokerAdapter(ABC):
    """Broker-facing operations; implementations must be non-destructive when not configured."""

    name: str = "abstract"

    @abstractmethod
    def connect(self) -> None:
        """Establish session (no-op for paper)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down session."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return (ok, message)."""

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Best-effort account snapshot."""

    def get_account_info(self) -> dict[str, Any]:
        """Structured account fields + raw broker payload (read-only helper)."""
        s = self.get_account_state()
        out: dict[str, Any] = {
            "equity": s.equity,
            "cash": s.cash,
            "margin_used": s.margin_used,
            "currency": s.currency,
        }
        if s.raw:
            out["raw"] = dict(s.raw)
        return out

    def get_balance(self) -> float | None:
        return self.get_account_state().cash

    def get_equity(self) -> float | None:
        return self.get_account_state().equity

    @abstractmethod
    def get_open_positions(self) -> list[PositionSnapshot]:
        ...

    @abstractmethod
    def get_open_orders(self) -> list[dict[str, Any]]:
        """Opaque broker order dicts for debugging."""

    @abstractmethod
    def submit_order(self, req: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> OrderResult:
        ...

    @abstractmethod
    def close_position(self, req: PositionCloseRequest) -> OrderResult:
        ...

    def get_recent_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        count: int,
        include_incomplete: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Optional live-market OHLCV fetch hook.

        Adapters that support market data should override this to return canonical
        rows with keys ``timestamp, open, high, low, close, volume``.
        """
        logger.warning(
            "adapter_get_recent_bars_not_supported adapter=%s symbol=%s timeframe=%s",
            self.name,
            symbol,
            timeframe,
        )
        return []
