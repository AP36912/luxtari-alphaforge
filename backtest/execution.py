"""
Execution logic: fill prices (spread + slippage), commission, and inventory updates.

Separated from **signal generation** — callers pass targets; this module only maps
targets to fills at the **next bar open** with explicit cost rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtest.config import ExecutionCosts, PositionSizing


def buy_fill_price(open_px: float, costs: ExecutionCosts) -> float:
    """Pay up to lift the offer: open * (1 + half_spread + slippage)."""
    return float(open_px * (1.0 + costs.half_spread + costs.slippage))


def sell_fill_price(open_px: float, costs: ExecutionCosts) -> float:
    """Hit the bid: open * (1 - half_spread - slippage)."""
    return float(open_px * (1.0 - costs.half_spread - costs.slippage))


def _commission(notional: float, costs: ExecutionCosts) -> float:
    return float(abs(notional) * costs.commission_rate)


def target_shares(
    signal: float,
    equity_at_decision: float,
    open_exec: float,
    sizing: PositionSizing,
) -> float:
    """
    Shares to hold after rebalance. Uses equity at **previous bar close** and execution **open**.

    No lookahead: ``equity_at_decision`` must not use ``open_exec`` or any bar >= execution bar.
    """
    if abs(signal) < 1e-15 or equity_at_decision <= 0 or open_exec <= 0:
        return 0.0
    if sizing.mode == "fixed_shares":
        return float(signal * sizing.fixed_shares)
    notional = signal * sizing.fraction_of_equity * equity_at_decision
    return float(notional / open_exec)


@dataclass
class TradeLeg:
    """One closed lot (for win rate / profit factor)."""

    bar: int
    pnl: float


def rebalance_to_target(
    *,
    bar_index: int,
    open_px: float,
    position: float,
    avg_entry: float,
    cash: float,
    target_shares_: float,
    costs: ExecutionCosts,
) -> tuple[float, float, float, list[TradeLeg]]:
    """
    Move from ``position`` to ``target_shares_`` using one bar's open.

    Uses **average-cost** inventory. Returns
    ``(new_position, new_avg_entry, new_cash, realized_trade_legs)``.
    """
    realized: list[TradeLeg] = []
    pos = float(position)
    avg = float(avg_entry)
    c = float(cash)
    target = float(target_shares_)
    tol = 1e-12

    def pay_commission(notional: float) -> None:
        nonlocal c
        c -= _commission(notional, costs)

    delta = target - pos

    # 1) Cover short (buy) while we need to increase position toward target
    while pos < -tol and delta > tol:
        cover = float(min(-pos, delta))
        fill = buy_fill_price(open_px, costs)
        notional = cover * fill
        pnl = (avg - fill) * cover
        c -= notional
        pay_commission(notional)
        realized.append(TradeLeg(bar=bar_index, pnl=pnl))
        pos += cover
        delta = target - pos
        if pos > -tol:
            pos = 0.0
            avg = 0.0
            delta = target - pos

    # 2) Sell long while we need to decrease position
    while pos > tol and delta < -tol:
        sell_qty = float(min(pos, -delta))
        fill = sell_fill_price(open_px, costs)
        notional = sell_qty * fill
        pnl = (fill - avg) * sell_qty
        c += notional
        pay_commission(notional)
        realized.append(TradeLeg(bar=bar_index, pnl=pnl))
        pos -= sell_qty
        delta = target - pos
        if pos < tol:
            pos = 0.0
            avg = 0.0
            delta = target - pos

    # 3) Buy (long or add long)
    if delta > tol:
        dq = delta
        fill = buy_fill_price(open_px, costs)
        notional = dq * fill
        c -= notional
        pay_commission(notional)
        if pos > tol:
            avg = (pos * avg + dq * fill) / (pos + dq)
        elif pos < -tol:
            raise RuntimeError("inconsistent state: short left after cover loop")
        else:
            avg = fill
        pos += dq

    # 4) Sell / open short
    elif delta < -tol:
        dq = -delta
        fill = sell_fill_price(open_px, costs)
        notional = dq * fill
        c += notional
        pay_commission(notional)
        if pos < -tol:
            avg = (abs(pos) * avg + dq * fill) / (abs(pos) + dq)
        elif pos > tol:
            raise RuntimeError("inconsistent state: long left after sell loop")
        else:
            avg = fill
        pos -= dq

    return pos, avg, c, realized
