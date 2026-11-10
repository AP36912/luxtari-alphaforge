"""
Atomic-ish persistence for paper/live service state between process restarts.

Uses a JSON document plus optional sidecar paths; not a substitute for a WAL.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.config import ExecutionAssumptions

from live.paper_broker import PaperBroker, PaperBrokerConfig, PaperPosition
from live.portfolio_state import PortfolioState
from live.live_types import LiveSignal, RegimeSnapshot


def _ts_iso(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.isoformat()
    return t.isoformat()


def _ts_parse(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


@dataclass
class LivePersistedState:
    version: int = 1
    last_processed_timestamp: str | None = None
    ohlcv_tail_records: list[dict[str, Any]] = field(default_factory=list)
    max_tail_bars: int = 5_000
    paper: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    latest_signal: dict[str, Any] | None = None
    model_primary_path: str | None = None
    model_meta_path: str | None = None
    feature_columns: list[str] = field(default_factory=list)


def live_signal_to_dict(s: LiveSignal) -> dict[str, Any]:
    return {
        "timestamp": _ts_iso(s.timestamp),
        "symbol": s.symbol,
        "direction": s.direction,
        "confidence": s.confidence,
        "primary_class": s.primary_class,
        "meta_class": s.meta_class,
        "regime": {
            "session": s.regime.session,
            "vol_state": s.regime.vol_state,
            "structure": s.regime.structure,
            "trading_allowed": s.regime.trading_allowed,
        },
    }


def live_signal_from_dict(d: dict[str, Any]) -> LiveSignal:
    r = d["regime"]
    return LiveSignal(
        timestamp=_ts_parse(str(d["timestamp"])),
        symbol=str(d["symbol"]),
        direction=int(d["direction"]),
        confidence=float(d["confidence"]),
        regime=RegimeSnapshot(
            session=str(r["session"]),
            vol_state=str(r["vol_state"]),
            structure=str(r["structure"]),
            trading_allowed=bool(r["trading_allowed"]),
        ),
        primary_class=int(d["primary_class"]),
        meta_class=None if d.get("meta_class") is None else int(d["meta_class"]),
    )


def paper_broker_to_dict(broker: PaperBroker) -> dict[str, Any]:
    st = broker.state
    pos = broker.open_position()
    cfg = broker.config
    return {
        "assumptions": cfg.assumptions.to_dict(),
        "contract_size": cfg.contract_size,
        "stop_loss_pct": cfg.stop_loss_pct,
        "take_profit_pct": cfg.take_profit_pct,
        "initial_cash_config": cfg.initial_cash,
        "cash": st.cash,
        "realized_pnl": st.realized_pnl,
        "pending_target_direction": st.pending_target_direction,
        "positions": []
        if pos is None
        else [
            {
                "position_id": pos.position_id,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "entry_time": _ts_iso(pos.entry_time),
                "entry_commission": pos.entry_commission,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }
        ],
    }


def paper_broker_apply_dict(broker: PaperBroker, d: dict[str, Any]) -> None:
    ass = ExecutionAssumptions.from_dict(d.get("assumptions"))
    sl = d.get("stop_loss_pct")
    tp = d.get("take_profit_pct")
    broker._cfg = PaperBrokerConfig(  # noqa: SLF001 — persistence restore
        assumptions=ass,
        initial_cash=float(d.get("initial_cash_config", broker.config.initial_cash)),
        contract_size=float(d.get("contract_size", broker.config.contract_size)),
        stop_loss_pct=None if sl is None else float(sl),
        take_profit_pct=None if tp is None else float(tp),
    )
    st = broker.state
    st.cash = float(d["cash"])
    st.realized_pnl = float(d.get("realized_pnl", 0.0))
    st.pending_target_direction = d.get("pending_target_direction")
    st.positions.clear()
    st.fills.clear()
    for p in d.get("positions") or []:
        st.positions.append(
            PaperPosition(
                position_id=str(p["position_id"]),
                side=p["side"],  # type: ignore[arg-type]
                quantity=float(p["quantity"]),
                entry_price=float(p["entry_price"]),
                entry_time=_ts_parse(str(p["entry_time"])),
                entry_commission=float(p.get("entry_commission", 0.0)),
                stop_loss=None if p.get("stop_loss") is None else float(p["stop_loss"]),
                take_profit=None if p.get("take_profit") is None else float(p["take_profit"]),
            )
        )


class LiveStateStore:
    """Write whole state to ``path`` atomically via temp + replace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> LivePersistedState | None:
        if not self.path.is_file():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return LivePersistedState(
            version=int(raw.get("version", 1)),
            last_processed_timestamp=raw.get("last_processed_timestamp"),
            ohlcv_tail_records=list(raw.get("ohlcv_tail_records") or []),
            max_tail_bars=int(raw.get("max_tail_bars", 5000)),
            paper=dict(raw.get("paper") or {}),
            portfolio=dict(raw.get("portfolio") or {}),
            latest_signal=raw.get("latest_signal"),
            model_primary_path=raw.get("model_primary_path"),
            model_meta_path=raw.get("model_meta_path"),
            feature_columns=list(raw.get("feature_columns") or []),
        )

    def save(self, state: LivePersistedState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        fd, tmp = tempfile.mkstemp(
            suffix=".json",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @staticmethod
    def ohlcv_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    @staticmethod
    def ohlcv_to_records(df: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
        tail = df.tail(max_rows).copy()
        tail["timestamp"] = tail["timestamp"].apply(lambda t: _ts_iso(pd.Timestamp(t)))
        return tail.to_dict(orient="records")
