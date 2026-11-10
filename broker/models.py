"""
Typed payloads for the execution gateway and broker adapters (Phase 17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import pandas as pd

OrderSide = Literal["buy", "sell", "flat"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]


class ExecutionDecisionStatus(str, Enum):
    """Terminal categories for a gateway pass."""

    EXECUTED = "executed"
    """Paper or broker path ran (fills may still be empty if flat target)."""
    DRY_RUN_LOGGED = "dry_run_logged"
    BLOCKED_VALIDATION = "blocked_validation"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_MODE = "blocked_mode"
    AWAITING_MANUAL_APPROVAL = "awaiting_manual_approval"
    BROKER_REJECTED = "broker_rejected"
    BROKER_ERROR = "broker_error"
    NOOP_RISK = "noop_risk_not_approved"
    READONLY_NO_SUBMIT = "readonly_no_submit"
    """Connected read-only path: account/positions observed, no broker orders."""
    SHADOW_WOULD_SUBMIT = "shadow_would_submit"
    """Live/demo intent validated but adapter submit suppressed (dry-run shadow)."""


@dataclass
class OrderRequest:
    """Normalized intent produced at the gateway (bar-aligned for paper path)."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    timestamp: pd.Timestamp
    execution_mode: str
    target_direction: int
    """Original signal direction {-1,0,+1} for paper broker compatibility."""
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    limit_price: float | None = None
    stop_price: float | None = None
    strategy_id: str | None = None
    run_id: str | None = None
    model_version: str | None = None
    regime_session: str | None = None
    regime_vol_state: str | None = None
    regime_structure: str | None = None
    trading_allowed: bool | None = None
    signal_confidence: float | None = None
    primary_class: int | None = None
    client_order_id: str | None = None


@dataclass
class OrderResult:
    timestamp_utc: str
    success: bool
    broker_order_id: str | None = None
    message: str = ""
    rejection_reason: str | None = None
    paper_fills_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CancelRequest:
    symbol: str
    broker_order_id: str
    client_order_id: str | None = None
    timestamp: pd.Timestamp | None = None


@dataclass
class PositionCloseRequest:
    symbol: str
    quantity: float | None = None
    """None = close full position."""
    timestamp: pd.Timestamp | None = None
    broker_position_id: str | None = None
    """When set (broker position ticket), close this ticket directly instead of symbol lookup (Phase 19.5)."""


@dataclass
class AccountState:
    equity: float | None = None
    cash: float | None = None
    margin_used: float | None = None
    currency: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    quantity: float
    avg_price: float | None = None
    unrealized_pnl: float | None = None
    broker_position_id: str | None = None


@dataclass
class ApprovalDecision:
    required: bool
    status: Literal["not_required", "pending", "approved", "denied"]
    approval_id: str | None = None
    path: str | None = None
    """Filesystem path when pending file was written."""


@dataclass
class ExecutionDecision:
    """Structured outcome from :class:`broker.gateway.ExecutionGateway`."""

    status: ExecutionDecisionStatus
    mode: str
    order_request: OrderRequest | None = None
    validation_errors: list[str] = field(default_factory=list)
    safety_reasons: list[str] = field(default_factory=list)
    approval: ApprovalDecision | None = None
    order_result: OrderResult | None = None
    audit_record_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        return self.status == ExecutionDecisionStatus.EXECUTED


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
