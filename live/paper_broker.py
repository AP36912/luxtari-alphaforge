"""
Paper execution engine — no real broker connectivity (Phase 16).

Uses :class:`backtest.config.ExecutionAssumptions` for parity with :class:`backtest.engine.BacktestEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from backtest.config import ExecutionAssumptions, PositionSizing
from backtest.execution import buy_fill_price, sell_fill_price

Side = Literal["flat", "long", "short"]


@dataclass
class PaperBrokerConfig:
    assumptions: ExecutionAssumptions = field(default_factory=ExecutionAssumptions)
    sizing: PositionSizing = field(default_factory=PositionSizing)
    initial_cash: float = 100_000.0
    contract_size: float = 1.0
    """Units per unit signal (e.g. ounces per +1)."""
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class PaperPosition:
    position_id: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_commission: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class PaperFill:
    timestamp: pd.Timestamp
    side: Side
    quantity: float
    price: float
    commission: float
    reason: str
    realized_pnl: float | None = None
    """Set on closing legs (increment to book realized PnL for that fill)."""


@dataclass
class PaperBrokerState:
    cash: float
    positions: list[PaperPosition] = field(default_factory=list)
    fills: list[PaperFill] = field(default_factory=list)
    realized_pnl: float = 0.0
    pending_target_direction: int | None = None


def _commission_dollars(notional: float, rate: float) -> float:
    return float(abs(notional) * rate)


class PaperBroker:
    """
    Single net position. Target direction ``{-1, 0, +1}`` sets side; magnitude is
    ``abs(direction)`` or 1 when non-zero.

    Stops run before discretionary targets for the bar. Fill prices use the same
    fractional spread/slippage model as the backtest via :func:`buy_fill_price` /
    :func:`sell_fill_price`.
    """

    def __init__(self, cfg: PaperBrokerConfig) -> None:
        self._cfg = cfg
        self._state = PaperBrokerState(cash=cfg.initial_cash)
        self.last_close: float | None = None

    @property
    def state(self) -> PaperBrokerState:
        return self._state

    def reset(self) -> None:
        self._state = PaperBrokerState(cash=self._cfg.initial_cash)
        self.last_close = None

    def open_position(self) -> PaperPosition | None:
        return self._state.positions[0] if self._state.positions else None

    @property
    def config(self) -> PaperBrokerConfig:
        return self._cfg

    @property
    def assumptions(self) -> ExecutionAssumptions:
        return self._cfg.assumptions

    def mark_to_market(self, last_close: float) -> float:
        pos = self.open_position()
        if pos is None:
            return 0.0
        q = pos.quantity
        if pos.side == "long":
            return q * (last_close - pos.entry_price) * self._cfg.contract_size
        return q * (pos.entry_price - last_close) * self._cfg.contract_size

    def equity(self, last_close: float) -> float:
        """Account equity: cash plus the signed market value of the open position."""
        pos = self.open_position()
        if pos is None:
            return float(self._state.cash)
        market_value = pos.quantity * float(last_close) * self._cfg.contract_size
        if pos.side == "short":
            market_value = -market_value
        return float(self._state.cash + market_value)

    def step(
        self,
        timestamp: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        target_direction: int,
    ) -> list[PaperFill]:
        self.last_close = float(close)
        fills: list[PaperFill] = []

        if self._cfg.assumptions.fill_basis == "next_bar_open" and self._state.pending_target_direction is not None:
            fills.extend(
                self._apply_target(int(self._state.pending_target_direction), timestamp, open_)
            )
            self._state.pending_target_direction = None

        fills.extend(self._check_stops(timestamp, high, low))

        td = int(target_direction)
        if self._cfg.assumptions.fill_basis == "next_bar_open":
            self._state.pending_target_direction = td
        else:
            fills.extend(self._apply_target(td, timestamp, close))

        return fills

    def _apply_target(self, target: int, ts: pd.Timestamp, ref_mid: float) -> list[PaperFill]:
        out: list[PaperFill] = []
        pos = self.open_position()
        cur = 0
        if pos is not None:
            cur = 1 if pos.side == "long" else -1

        if target == cur:
            return out

        if pos is not None:
            out.append(self._close_position(ts, ref_mid, "signal_rebalance"))

        if target == 0:
            return out

        side: Side = "long" if target > 0 else "short"
        if self._cfg.sizing.mode != "fixed_shares":
            raise ValueError(
                "PaperBroker parity supports only sizing.mode='fixed_shares'; "
                "fraction_of_equity is unsupported"
            )
        qty = float(abs(target)) * float(self._cfg.sizing.fixed_shares)
        out.append(self._open_position_side(ts, side, qty, ref_mid))
        return out

    def _open_position_side(self, ts: pd.Timestamp, side: Side, qty: float, ref: float) -> PaperFill:
        costs = self._cfg.assumptions.costs
        if side == "long":
            px = buy_fill_price(ref, costs)
        else:
            px = sell_fill_price(ref, costs)
        notional = qty * px * self._cfg.contract_size
        comm = _commission_dollars(notional, costs.commission_rate)
        if side == "long":
            self._state.cash -= notional + comm
        else:
            self._state.cash += notional - comm
        sl, tp = self._stops_for_entry(side, px)
        pid = str(uuid4())[:12]
        self._state.positions.append(
            PaperPosition(
                position_id=pid,
                side=side,
                quantity=qty,
                entry_price=px,
                entry_time=ts,
                entry_commission=comm,
                stop_loss=sl,
                take_profit=tp,
            )
        )
        fl = PaperFill(ts, side, qty, px, comm, "open", realized_pnl=None)
        self._state.fills.append(fl)
        return fl

    def _close_position(self, ts: pd.Timestamp, ref: float, reason: str) -> PaperFill:
        pos = self.open_position()
        if pos is None:
            raise RuntimeError("close without position")
        qty = pos.quantity
        costs = self._cfg.assumptions.costs
        if pos.side == "long":
            px = sell_fill_price(ref, costs)
        else:
            px = buy_fill_price(ref, costs)
        notional = qty * px * self._cfg.contract_size
        comm = _commission_dollars(notional, costs.commission_rate)
        if pos.side == "long":
            self._state.cash += notional - comm
            pnl = qty * (px - pos.entry_price) * self._cfg.contract_size - pos.entry_commission - comm
        else:
            self._state.cash -= notional + comm
            pnl = qty * (pos.entry_price - px) * self._cfg.contract_size - pos.entry_commission - comm
        self._state.realized_pnl += pnl
        self._state.positions.clear()
        fl = PaperFill(ts, pos.side, qty, px, comm, reason, realized_pnl=pnl)
        self._state.fills.append(fl)
        return fl

    def _stop_exit_ref(self, pos: PaperPosition, high: float, low: float, *, is_stop: bool) -> float:
        """Reference price for exit before applying sell/buy spread."""
        pol = self._cfg.assumptions.stop_target_fill_policy
        if pos.side == "long":
            if is_stop:
                lvl = float(pos.stop_loss) if pos.stop_loss is not None else low
                if pol == "bar_extreme":
                    return min(lvl, low)
                return lvl
            lvl = float(pos.take_profit) if pos.take_profit is not None else high
            if pol == "bar_extreme":
                return max(lvl, high)
            return lvl
        if is_stop:
            lvl = float(pos.stop_loss) if pos.stop_loss is not None else high
            if pol == "bar_extreme":
                return max(lvl, high)
            return lvl
        lvl = float(pos.take_profit) if pos.take_profit is not None else low
        if pol == "bar_extreme":
            return min(lvl, low)
        return lvl

    def _check_stops(self, ts: pd.Timestamp, high: float, low: float) -> list[PaperFill]:
        out: list[PaperFill] = []
        pos = self.open_position()
        if pos is None:
            return out
        if pos.side == "long":
            if pos.stop_loss is not None and low <= pos.stop_loss:
                ref = self._stop_exit_ref(pos, high, low, is_stop=True)
                out.append(self._close_position(ts, ref, "stop_loss"))
            elif pos.take_profit is not None and high >= pos.take_profit:
                ref = self._stop_exit_ref(pos, high, low, is_stop=False)
                out.append(self._close_position(ts, ref, "take_profit"))
        else:
            if pos.stop_loss is not None and high >= pos.stop_loss:
                ref = self._stop_exit_ref(pos, high, low, is_stop=True)
                out.append(self._close_position(ts, ref, "stop_loss"))
            elif pos.take_profit is not None and low <= pos.take_profit:
                ref = self._stop_exit_ref(pos, high, low, is_stop=False)
                out.append(self._close_position(ts, ref, "take_profit"))
        return out

    def _stops_for_entry(self, side: Side, entry: float) -> tuple[float | None, float | None]:
        sl_pct = self._cfg.stop_loss_pct
        tp_pct = self._cfg.take_profit_pct
        if sl_pct is None and tp_pct is None:
            return None, None
        if side == "long":
            sl = entry * (1.0 - sl_pct) if sl_pct is not None else None
            tp = entry * (1.0 + tp_pct) if tp_pct is not None else None
        else:
            sl = entry * (1.0 + sl_pct) if sl_pct is not None else None
            tp = entry * (1.0 - tp_pct) if tp_pct is not None else None
        return sl, tp

    def portfolio_snapshot(self, last_close: float) -> dict[str, Any]:
        pos = self.open_position()
        net = 0.0
        if pos is not None:
            net = pos.quantity if pos.side == "long" else -pos.quantity
        return {
            "cash": self._state.cash,
            "equity": self.equity(last_close),
            "realized_pnl": self._state.realized_pnl,
            "unrealized_pnl": self.mark_to_market(last_close),
            "open_position_count": 1 if pos is not None else 0,
            "net_exposure_shares": net,
            "position": None
            if pos is None
            else {
                "id": pos.position_id,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            },
        }


def paper_broker_config_from_backtest(bt: Any, *, contract_size: float = 1.0) -> PaperBrokerConfig:
    """Build paper config for the explicitly supported fixed-share parity scope."""
    from backtest.config import BacktestConfig

    if not isinstance(bt, BacktestConfig):
        raise TypeError("expected BacktestConfig")
    if bt.sizing.mode != "fixed_shares":
        raise ValueError(
            "paper/backtest parity does not support fraction_of_equity sizing; use fixed_shares"
        )
    return PaperBrokerConfig(
        assumptions=bt.assumptions,
        sizing=bt.sizing,
        initial_cash=float(bt.initial_capital),
        contract_size=float(contract_size),
        stop_loss_pct=bt.stop_loss_pct,
        take_profit_pct=bt.take_profit_pct,
    )
