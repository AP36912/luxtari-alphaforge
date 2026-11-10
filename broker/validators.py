"""Order/config validators — public paper/sim edition."""

from __future__ import annotations

from typing import Any, Literal

from broker.mode import ExecutionMode, parse_execution_mode
from broker.models import OrderRequest

OrderSide = Literal["buy", "sell", "flat"]

_VALID_SIDES: set[str] = {"buy", "sell", "flat"}
_VALID_TYPES: set[str] = {"market", "limit", "stop", "stop_limit"}


def validate_order_request(req: OrderRequest, cfg: Any | None = None) -> list[str]:
    """Return list of error codes/messages; empty means OK."""
    errors: list[str] = []
    if not str(req.symbol or "").strip():
        errors.append("missing_symbol")
    if req.side not in _VALID_SIDES:
        errors.append(f"invalid_side:{req.side}")
    if req.order_type not in _VALID_TYPES:
        errors.append(f"invalid_order_type:{req.order_type}")
    if req.quantity < 0:
        errors.append("negative_quantity")
    if req.side != "flat" and req.quantity <= 0:
        errors.append("non_flat_requires_positive_quantity")
    if req.order_type == "limit" and req.limit_price is None:
        errors.append("limit_order_requires_limit_price")
    if req.order_type == "stop" and req.stop_price is None:
        errors.append("stop_order_requires_stop_price")

    if cfg is not None:
        try:
            mode = parse_execution_mode(getattr(cfg, "execution_mode", "paper"))
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if mode not in {ExecutionMode.PAPER, ExecutionMode.DRY_RUN, ExecutionMode.SIM}:
                errors.append(f"execution_mode_forbidden:{mode.value}")
        for forbidden in ("broker_live_enabled", "broker_submit_enabled"):
            if hasattr(cfg, forbidden):
                errors.append(f"{forbidden}_forbidden_in_public_edition")
    return errors


def direction_to_side(direction: int) -> OrderSide:
    if direction > 0:
        return "buy"
    if direction < 0:
        return "sell"
    return "flat"


def quantity_for_direction(direction: int, contract_size: float = 1.0) -> float:
    if direction == 0:
        return 0.0
    return float(abs(direction)) if abs(direction) > 1 else 1.0
