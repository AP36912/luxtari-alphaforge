"""
Aggregated paper account metrics for risk checks and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


def _utc_day(d: datetime | pd.Timestamp) -> date:
    if isinstance(d, pd.Timestamp):
        if d.tzinfo is not None:
            return d.tz_convert("UTC").date()
        return d.date()
    if d.tzinfo is not None:
        return d.astimezone(timezone.utc).date()
    return d.date()


@dataclass
class PortfolioState:
    """
    Snapshot for :class:`live.risk_gate.RiskGate`.

    Drawdown trackers are updated via :meth:`on_equity_tick`.
    """

    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    open_position_count: int
    last_price: float
    contract_size: float = 1.0
    net_exposure_shares: float = 0.0
    """Signed size of open position (+ long, - short)."""
    peak_equity_today: float = 0.0
    day_anchor: date | None = None
    session_peak_equity: dict[str, float] = field(default_factory=dict)
    last_session_tag: str | None = None

    def on_equity_tick(self, ts: pd.Timestamp | datetime, equity: float, session_tag: str) -> None:
        """Call after each mark-to-market update to refresh drawdown peaks."""
        day = _utc_day(ts)
        if self.day_anchor != day:
            self.day_anchor = day
            self.peak_equity_today = equity
        self.peak_equity_today = max(self.peak_equity_today, equity)

        if self.last_session_tag != session_tag:
            self.last_session_tag = session_tag
            self.session_peak_equity[session_tag] = equity
        self.session_peak_equity[session_tag] = max(
            self.session_peak_equity.get(session_tag, equity), equity
        )

    def current_daily_drawdown(self) -> float:
        if self.peak_equity_today <= 0:
            return 0.0
        return (self.peak_equity_today - self.equity) / self.peak_equity_today

    def current_session_drawdown(self, session_tag: str) -> float:
        peak = self.session_peak_equity.get(session_tag, self.equity)
        if peak <= 0:
            return 0.0
        return (peak - self.equity) / peak

    def signal_maintains_or_reduces(self, direction: int) -> bool:
        """
        True when the intent does not require *additional* capacity (flatten, hold, flip).
        Opening from flat while at the concurrent cap is False.
        """
        if direction == 0:
            return True
        if self.open_position_count == 0:
            return False
        same = (self.net_exposure_shares > 0 and direction > 0) or (
            self.net_exposure_shares < 0 and direction < 0
        )
        opposite = self.net_exposure_shares * direction < 0
        return same or opposite

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "open_position_count": self.open_position_count,
            "last_price": self.last_price,
            "contract_size": self.contract_size,
            "net_exposure_shares": self.net_exposure_shares,
            "peak_equity_today": self.peak_equity_today,
            "day_anchor": self.day_anchor.isoformat() if self.day_anchor else None,
            "session_peak_equity": dict(self.session_peak_equity),
            "last_session_tag": self.last_session_tag,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortfolioState:
        da = d.get("day_anchor")
        return cls(
            equity=float(d["equity"]),
            cash=float(d["cash"]),
            unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            open_position_count=int(d.get("open_position_count", 0)),
            last_price=float(d.get("last_price", 0.0)),
            contract_size=float(d.get("contract_size", 1.0)),
            net_exposure_shares=float(d.get("net_exposure_shares", 0.0)),
            peak_equity_today=float(d.get("peak_equity_today", d.get("equity", 0.0))),
            day_anchor=date.fromisoformat(da) if da else None,
            session_peak_equity={k: float(v) for k, v in (d.get("session_peak_equity") or {}).items()},
            last_session_tag=d.get("last_session_tag"),
        )
