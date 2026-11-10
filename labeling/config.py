"""
Labeling configuration: triple barriers and meta-label rules.

Barriers are evaluated **only on bars strictly after** the decision index
(``t+1 .. t+max_bars``) so features at ``t`` remain causally prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["long", "short"]
VolatilityScale = Literal["returns_std", "atr"]
MetaLabelMode = Literal["profit_first", "decisive", "avoid_loss"]


@dataclass
class TripleBarrierConfig:
    """
    Triple barrier method (profit / stop / vertical time cap).

    Labels (for the chosen ``position_side``):
        * ``1`` — profit barrier touched first
        * ``-1`` — stop barrier touched first
        * ``0`` — vertical barrier (``max_bars``) reached without either
        * ``nan`` — not enough future bars to complete the window
    """

    max_bars: int
    """Maximum number of **future** bars to scan (vertical barrier)."""

    position_side: PositionSide = "long"
    """``long``: profit above entry, stop below. ``short``: profit below, stop above."""

    # Fixed fractional barriers (used when ``use_volatility_scaling`` is False)
    profit_pct: float = 0.02
    """Take-profit distance as a fraction of entry (e.g. 0.02 = 2%)."""
    stop_pct: float = 0.02
    """Stop distance as a fraction of entry."""

    use_volatility_scaling: bool = False
    """If True, scale barriers by a volatility estimate at the decision bar."""

    volatility_window: int = 20
    """Lookback for volatility (returns std or ATR period)."""
    volatility_scale: VolatilityScale = "returns_std"
    profit_vol_multiple: float = 1.0
    """Profit barrier width = ``entry * profit_vol_multiple * vol`` (return space)."""
    stop_vol_multiple: float = 1.0
    """Stop barrier width = ``entry * stop_vol_multiple * vol``."""

    entry_price_column: Literal["close", "open"] = "close"
    """Price at the decision bar used as barrier anchor."""

    atr_period: int = 14
    """ATR period when ``volatility_scale == \"atr\"`` (Wilder EWM on true range)."""


@dataclass
class MetaLabelConfig:
    """
    Secondary (meta) label derived from the primary triple-barrier outcome.

    Modes:
        * ``profit_first`` — ``1`` if primary ``== 1`` else ``0`` (hit take-profit first).
        * ``decisive`` — ``1`` if primary in ``{-1, 1}`` else ``0`` (any barrier before time).
        * ``avoid_loss`` — ``1`` if primary ``!= -1`` else ``0`` (did not stop out first).
    """

    mode: MetaLabelMode = "profit_first"
