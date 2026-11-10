"""
Backtest configuration: trading costs, execution assumptions, and position sizing.

``ExecutionAssumptions`` is shared with :mod:`live.paper_broker` (Phase 16.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ExecutionCosts:
    """
    Proportional costs applied at each fill (fractions of the reference price).

    - ``half_spread``: half the bid–ask spread as a fraction of mid, paid per fill.
    - ``slippage``: extra adverse movement as a fraction (model uncertainty / impact).
    - ``commission_rate``: fraction of notional charged on each fill (buy and sell).
    """

    half_spread: float = 0.0
    slippage: float = 0.0
    commission_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, v in (
            ("half_spread", self.half_spread),
            ("slippage", self.slippage),
            ("commission_rate", self.commission_rate),
        ):
            if v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")


FillBasis = Literal["bar_close", "next_bar_open"]
StopTargetFillPolicy = Literal["bar_extreme", "stop_level"]


def validate_fill_basis(value: str) -> FillBasis:
    if value not in ("bar_close", "next_bar_open"):
        raise ValueError(
            f"fill_basis must be 'bar_close' or 'next_bar_open', got {value!r}"
        )
    return value  # type: ignore[return-value]


def validate_stop_policy(value: str) -> StopTargetFillPolicy:
    if value not in ("bar_extreme", "stop_level"):
        raise ValueError(
            f"stop_target_fill_policy must be 'bar_extreme' or 'stop_level', got {value!r}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class ExecutionAssumptions:
    """
    Shared semantics for :class:`BacktestEngine` and :class:`live.paper_broker.PaperBroker`.
    """

    fill_basis: FillBasis = "next_bar_open"
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    stop_target_fill_policy: StopTargetFillPolicy = "bar_extreme"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_basis", validate_fill_basis(str(self.fill_basis)))
        object.__setattr__(
            self,
            "stop_target_fill_policy",
            validate_stop_policy(str(self.stop_target_fill_policy)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_basis": self.fill_basis,
            "stop_target_fill_policy": self.stop_target_fill_policy,
            "costs": {
                "half_spread": self.costs.half_spread,
                "slippage": self.costs.slippage,
                "commission_rate": self.costs.commission_rate,
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ExecutionAssumptions:
        if not raw:
            return cls()
        costs_d = raw.get("costs") or {}
        ckw = {k: float(costs_d[k]) for k in ("half_spread", "slippage", "commission_rate") if k in costs_d}
        costs = ExecutionCosts(**ckw) if ckw else ExecutionCosts()
        return cls(
            fill_basis=validate_fill_basis(str(raw.get("fill_basis", "next_bar_open"))),
            costs=costs,
            stop_target_fill_policy=validate_stop_policy(
                str(raw.get("stop_target_fill_policy", "bar_extreme"))
            ),
        )


SizingMode = Literal["fixed_shares", "fraction_of_equity"]


@dataclass(frozen=True)
class PositionSizing:
    """
    How target position size is chosen from the signal and mark-to-market equity.

    ``fixed_shares``: target shares = ``signal * fixed_shares``.
    ``fraction_of_equity``: target notional = ``signal * fraction_of_equity * equity_at_decision``,
    shares = notional / execution reference price.
    """

    mode: SizingMode = "fixed_shares"
    fixed_shares: float = 1.0
    fraction_of_equity: float = 0.99

    def __post_init__(self) -> None:
        if self.mode == "fixed_shares" and self.fixed_shares <= 0:
            raise ValueError("fixed_shares must be positive")
        if self.mode == "fraction_of_equity":
            if not 0 < self.fraction_of_equity <= 1:
                raise ValueError("fraction_of_equity must be in (0, 1]")


@dataclass(frozen=True)
class BacktestConfig:
    """Top-level knobs for :class:`BacktestEngine`."""

    initial_capital: float = 100_000.0
    assumptions: ExecutionAssumptions = field(default_factory=ExecutionAssumptions)
    sizing: PositionSizing = field(default_factory=PositionSizing)
    bars_per_year: int = 252
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    @property
    def costs(self) -> ExecutionCosts:
        """Backward-compatible alias for ``assumptions.costs``."""
        return self.assumptions.costs

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.bars_per_year < 2:
            raise ValueError("bars_per_year must be >= 2")
        for name, value in (("stop_loss_pct", self.stop_loss_pct), ("take_profit_pct", self.take_profit_pct)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
