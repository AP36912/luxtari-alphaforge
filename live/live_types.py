"""
Shared types for the live / paper layer (no broker execution in this phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

FillPriceBasis = Literal["bar_close", "next_bar_open"]


@dataclass(frozen=True)
class RegimeSnapshot:
    """Current session and rule-based regime tags for one decision bar."""

    session: str
    vol_state: str
    structure: str
    trading_allowed: bool


@dataclass(frozen=True)
class LiveSignal:
    """Structured output after primary (and optional meta) inference."""

    timestamp: pd.Timestamp
    symbol: str
    direction: int
    """Signed target: -1 short, 0 flat, +1 long (maps from model classes when configured)."""
    confidence: float
    """Max predicted class probability when available, else 1.0."""
    regime: RegimeSnapshot
    primary_class: int
    meta_class: int | None = None
    raw_proba: list[float] | None = None


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class QuoteSnapshot:
    """Optional top-of-book or bar-derived execution quality hints."""

    mid: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    timestamp: datetime | None = None


@dataclass
class SchedulerStepResult:
    """One orchestrated live tick (deterministic given inputs)."""

    timestamp: pd.Timestamp | None
    signal: LiveSignal | None
    risk: RiskDecision | None
    executed: bool
    notes: list[str] = field(default_factory=list)
    execution_decision: Any | None = None
    """Populated when :class:`~live.scheduler.LiveScheduler` uses :class:`broker.gateway.ExecutionGateway`."""
    signal_diagnostics: dict[str, Any] | None = None
    """Phase 20.5: model→direction pipeline (optional; demo automation / diagnostics)."""
