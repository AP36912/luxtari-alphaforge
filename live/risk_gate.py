"""
Final pre-trade checks before paper (or future live) execution.

Pure functions + :class:`RiskGate` for dependency injection (portfolio, quotes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from live.live_types import LiveSignal, QuoteSnapshot, RiskDecision

if TYPE_CHECKING:
    from live.portfolio_state import PortfolioState


@dataclass
class RiskGateConfig:
    max_risk_per_trade_fraction: float = 0.02
    """Cap notional as fraction of equity (0–1)."""
    max_concurrent_positions: int = 1
    max_daily_drawdown: float = 0.2
    """Halt when drawdown from intra-day peak exceeds this fraction (0–1)."""
    max_session_drawdown: float | None = None
    """Optional drawdown cap within the current session tag (0–1)."""
    max_spread_bps: float | None = None
    """Reject when quote spread (bps) exceeds threshold; skipped if no quote."""
    require_regime_permission: bool = True
    """If True, block when ``signal.regime.trading_allowed`` is False."""


@dataclass
class RiskContext:
    signal: LiveSignal
    portfolio: PortfolioState
    quote: QuoteSnapshot | None = None
    now_utc: datetime | None = None


class RiskGate:
    def __init__(self, cfg: RiskGateConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> RiskGateConfig:
        return self._cfg

    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        reasons: list[str] = []
        s = ctx.signal
        p = ctx.portfolio

        if self._cfg.require_regime_permission and not s.regime.trading_allowed:
            reasons.append("regime_or_session_filter_blocks_trading")

        if s.direction != 0 and self._cfg.max_spread_bps is not None and ctx.quote is not None:
            if ctx.quote.spread_bps is not None and ctx.quote.spread_bps > self._cfg.max_spread_bps:
                reasons.append(f"spread_bps {ctx.quote.spread_bps:.2f} > max {self._cfg.max_spread_bps}")

        if s.direction != 0:
            notional = abs(s.direction) * p.last_price * p.contract_size
            max_notional = p.equity * self._cfg.max_risk_per_trade_fraction
            if notional > max_notional + 1e-9:
                reasons.append(
                    f"notional {notional:.2f} exceeds max_risk_fraction "
                    f"({self._cfg.max_risk_per_trade_fraction:.4f} * equity)"
                )

        if s.direction != 0:
            if self._cfg.max_concurrent_positions <= 0:
                reasons.append("max_concurrent_positions_disabled")
            elif (
                p.open_position_count >= self._cfg.max_concurrent_positions
                and not p.signal_maintains_or_reduces(s.direction)
            ):
                reasons.append("max_concurrent_positions")

        dd_day = p.current_daily_drawdown()
        if dd_day > self._cfg.max_daily_drawdown:
            reasons.append(f"daily_drawdown {dd_day:.4f} > max {self._cfg.max_daily_drawdown}")

        if self._cfg.max_session_drawdown is not None:
            dd_sess = p.current_session_drawdown(s.regime.session)
            if dd_sess > self._cfg.max_session_drawdown:
                reasons.append(
                    f"session_drawdown {dd_sess:.4f} > max {self._cfg.max_session_drawdown}"
                )

        return RiskDecision(approved=len(reasons) == 0, reasons=reasons)
