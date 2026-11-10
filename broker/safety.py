from __future__ import annotations

"""Safety limits — public paper/sim edition.
Sanitized export replacement."""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from broker.mode import ExecutionMode, parse_execution_mode
from broker.models import OrderRequest

from live.portfolio_state import PortfolioState


@dataclass
class SafetyState:
    """Mutable runtime state for gateway safety (not persisted across processes unless wired)."""

    day_anchor: date | None = None
    equity_day_start: float | None = None
    trading_halted_daily_loss: bool = False
    last_submit_mono: dict[str, float] = field(default_factory=dict)
    """key -> monotonic time of last submission for cooldown."""


@dataclass
class SafetyEvaluation:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def _kill_switch_file_status(path: Path | None) -> tuple[bool, bool]:
    """Return ``(active, read_error)``; callers must fail closed on read errors.

    Uses ``Path.stat()`` rather than ``is_file()`` because pathlib commonly
    swallows permission errors into ``False`` for ``is_file()``.
    """
    if path is None:
        return False, False
    try:
        import stat as _stat

        return _stat.S_ISREG(path.stat().st_mode), False
    except FileNotFoundError:
        return False, False
    except OSError:
        return False, True


def evaluate_execution_safety(
    req: OrderRequest,
    cfg: Any,
    state: SafetyState,
    portfolio: PortfolioState,
    *,
    now_utc: datetime | None = None,
    signal_timestamp: pd.Timestamp | None = None,
    kill_switch_file_path: Path | None = None,
) -> SafetyEvaluation:
    """
    Apply kill switch, stale signal, daily loss halt, size/notional/concurrency limits,
    symbol lists, duplicate cooldown.
    ``cfg`` should be :class:`config.experiment_schema.ExecutionConfig`.
    """
    reasons: list[str] = []
    now = now_utc or datetime.now(timezone.utc)

    # ``kill_switch_enabled`` means file monitoring is configured; it is not a halt.
    if bool(getattr(cfg, "emergency_stop", False)):
        reasons.append("emergency_stop")
    kpath = kill_switch_file_path
    if kpath is None and getattr(cfg, "kill_switch_file_path", None):
        kpath = Path(str(cfg.kill_switch_file_path))
    kill_active, kill_read_error = _kill_switch_file_status(kpath)
    if kill_active:
        reasons.append("kill_switch_file_present")
    if kill_read_error:
        reasons.append("kill_switch_file_read_error")

    if signal_timestamp is not None and cfg.stale_signal_max_age_seconds is not None:
        sig_ts = pd.Timestamp(signal_timestamp)
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize("UTC")
        else:
            sig_ts = sig_ts.tz_convert("UTC")
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        age = float((now_ts - sig_ts).total_seconds())
        if age > float(cfg.stale_signal_max_age_seconds):
            reasons.append(f"stale_signal age_seconds={age:.1f}")

    # Daily loss tracking (reset on day change)
    bar_day = now.date()
    if state.day_anchor != bar_day:
        state.day_anchor = bar_day
        state.equity_day_start = float(portfolio.equity)
        state.trading_halted_daily_loss = False

    if state.trading_halted_daily_loss:
        reasons.append("trading_halted_max_daily_loss")

    if (
        cfg.max_daily_loss_fraction is not None
        and state.equity_day_start is not None
        and state.equity_day_start > 0
    ):
        dd = (state.equity_day_start - float(portfolio.equity)) / state.equity_day_start
        if dd >= float(cfg.max_daily_loss_fraction):
            state.trading_halted_daily_loss = True
            reasons.append(f"max_daily_loss_exceeded fraction={dd:.4f}")

    sym = req.symbol.upper()
    if cfg.symbol_allowlist:
        allowed = {s.upper() for s in cfg.symbol_allowlist}
        if sym not in allowed:
            reasons.append("symbol_not_on_allowlist")
    if cfg.symbol_denylist:
        denied = {s.upper() for s in cfg.symbol_denylist}
        if sym in denied:
            reasons.append("symbol_on_denylist")

    if req.side != "flat" and req.quantity > 0:
        if cfg.max_position_size is not None and req.quantity > float(cfg.max_position_size):
            reasons.append("max_position_size_exceeded")

        px = float(req.bar_close)
        notional = abs(req.quantity) * px * float(portfolio.contract_size)
        if cfg.max_notional is not None and notional > float(cfg.max_notional):
            reasons.append("max_notional_exceeded")

    if cfg.max_concurrent_positions is not None and int(cfg.max_concurrent_positions) <= 0:
        if req.target_direction != 0:
            reasons.append("execution_max_concurrent_positions_nonpositive")

    if (
        cfg.max_concurrent_positions is not None
        and int(cfg.max_concurrent_positions) > 0
        and req.target_direction != 0
        and req.quantity > 0
    ):
        mx = int(cfg.max_concurrent_positions)
        if int(portfolio.open_position_count) >= mx:
            reasons.append(
                f"max_concurrent_positions_exceeded open={portfolio.open_position_count} max={mx}"
            )

    return SafetyEvaluation(allowed=len(reasons) == 0, reasons=reasons)


def duplicate_cooldown_blocks(
    cfg: Any,
    state: SafetyState,
    key: str,
    *,
    now_mono: float,
) -> bool:
    if cfg.duplicate_order_cooldown_seconds <= 0:
        return False
    last = state.last_submit_mono.get(key)
    if last is None:
        return False
    return (now_mono - last) < float(cfg.duplicate_order_cooldown_seconds)


def mark_submit_cooldown(state: SafetyState, key: str, now_mono: float) -> None:
    state.last_submit_mono[key] = now_mono
