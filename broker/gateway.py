"""Fail-closed execution gateway for paper, simulation, and dry-run modes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from broker.demo_adapter import DemoBrokerAdapter
from broker.mode import ExecutionMode, normalize_execution_mode
from broker.models import OrderRequest, OrderResult, utc_now_iso
from broker.paper_adapter import PaperBrokerAdapter
from broker.safety import (
    SafetyState,
    duplicate_cooldown_blocks,
    evaluate_execution_safety,
    mark_submit_cooldown,
)
from live.paper_broker import PaperBroker, PaperBrokerConfig
from live.portfolio_state import PortfolioState


@dataclass
class GatewayConfig:
    execution_mode: str = "paper"
    initial_cash: float = 100_000.0
    contract_size: float = 1.0
    emergency_stop: bool = False
    kill_switch_file_path: str | None = None
    stale_signal_max_age_seconds: float | None = 300.0
    duplicate_order_cooldown_seconds: float = 0.0
    symbol_allowlist: list[str] = field(default_factory=list)
    symbol_denylist: list[str] = field(default_factory=list)
    max_position_size: float | None = None
    max_notional: float | None = None
    max_concurrent_positions: int | None = None
    max_daily_loss_fraction: float | None = None


class ExecutionGateway:
    """Route risk-approved bar intents without exposing a live-broker path."""

    def __init__(self, cfg: GatewayConfig | None = None):
        self.cfg = cfg or GatewayConfig()
        self.mode = normalize_execution_mode(self.cfg.execution_mode)
        if self.mode not in {ExecutionMode.PAPER, ExecutionMode.DRY_RUN, ExecutionMode.SIM}:
            raise ValueError(f"public execution gateway refuses mode={self.mode.value!r}")
        self.safety_state = SafetyState()
        if self.mode in {ExecutionMode.PAPER, ExecutionMode.SIM}:
            paper = PaperBroker(PaperBrokerConfig(
                initial_cash=float(self.cfg.initial_cash),
                contract_size=float(self.cfg.contract_size),
            ))
            self.adapter: Any = PaperBrokerAdapter(paper)
        else:
            self.adapter = DemoBrokerAdapter()

    @property
    def config(self) -> GatewayConfig:
        return self.cfg

    def connect(self) -> None:
        self.adapter.connect()

    def disconnect(self) -> None:
        self.adapter.disconnect()

    def health_check(self) -> tuple[bool, str]:
        return self.adapter.health_check()

    def _portfolio(self, req: OrderRequest) -> PortfolioState:
        account = self.adapter.get_account_state()
        positions = self.adapter.get_open_positions()
        cash = float(account.cash if account.cash is not None else self.cfg.initial_cash)
        equity = float(account.equity if account.equity is not None else cash)
        net = sum((1.0 if p.side == "long" else -1.0) * p.quantity for p in positions)
        return PortfolioState(
            equity=equity,
            cash=cash,
            unrealized_pnl=equity - cash,
            realized_pnl=0.0,
            open_position_count=len(positions),
            last_price=float(req.bar_close),
            contract_size=float(self.cfg.contract_size),
            net_exposure_shares=float(net),
        )

    def process_risk_approved_bar(
        self, req: OrderRequest, *, portfolio: PortfolioState | None = None
    ) -> OrderResult:
        """Evaluate public safety controls and execute one normalized bar intent."""
        if normalize_execution_mode(req.execution_mode) != self.mode:
            return OrderResult(
                timestamp_utc=utc_now_iso(), success=False,
                rejection_reason="request_execution_mode_mismatch",
                message="gateway_rejected",
            )
        port = portfolio or self._portfolio(req)
        evaluation = evaluate_execution_safety(
            req,
            self.cfg,
            self.safety_state,
            port,
            signal_timestamp=req.timestamp,
            kill_switch_file_path=(
                Path(self.cfg.kill_switch_file_path)
                if self.cfg.kill_switch_file_path else None
            ),
        )
        key = req.client_order_id or f"{req.symbol}:{req.target_direction}"
        now_mono = time.monotonic()
        if duplicate_cooldown_blocks(self.cfg, self.safety_state, key, now_mono=now_mono):
            evaluation.allowed = False
            evaluation.reasons.append("duplicate_order_cooldown")
        if not evaluation.allowed:
            return OrderResult(
                timestamp_utc=utc_now_iso(), success=False,
                rejection_reason="; ".join(evaluation.reasons),
                message="gateway_safety_rejected",
            )
        if self.mode in {ExecutionMode.PAPER, ExecutionMode.SIM}:
            result, _fills = self.adapter.execute_bar_intent(req)
        else:
            result = self.adapter.submit_order(req)
        if result.success:
            mark_submit_cooldown(self.safety_state, key, now_mono)
        return result
