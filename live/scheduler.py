"""
Single-threaded bar-by-bar orchestration for paper / live inference.

No network I/O inside the scheduler itself — inject :class:`live.data_feed.AbstractDataFeed`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from live.alerting import AlertSink, live_alert
from live.data_feed import merge_ohlcv_buffer
from live.feature_service import LiveFeatureService
from live.paper_broker import PaperBroker
from live.portfolio_state import PortfolioState
from live.risk_gate import RiskContext, RiskGate
from live.signal_service import SignalService
from live.state_store import (
    LivePersistedState,
    LiveStateStore,
    live_signal_to_dict,
    paper_broker_apply_dict,
    paper_broker_to_dict,
)
from live.live_types import LiveSignal, QuoteSnapshot, SchedulerStepResult

logger = logging.getLogger(__name__)



@dataclass
class LiveSchedulerConfig:
    symbol: str = "SYNTH"
    expected_bar_delta: pd.Timedelta | None = None
    persist_tail_bars: int = 2_000
    model_primary_path: str | None = None
    model_meta_path: str | None = None
    feature_columns: list[str] | None = None
    enable_bar_dedupe: bool = True
    """Skip broker execution when the decision bar timestamp matches the last persisted bar."""
    run_id: str | None = None
    """Propagated to execution gateway audit metadata when set."""
    signal_diagnostics_enabled: bool = False
    """When True, :meth:`step_once` attaches ``signal_diagnostics`` on the step result."""
    test_only_force_signal_direction: int | None = None
    """Phase 20.5 demo validation: replace model direction with {-1,0,+1}; None disables."""
    block_entries_trend_regime_downtrend: bool = False
    """When True, unsupported in public edition; scheduler fails closed if enabled."""
    trend_regime_atr_period: int = 14
    """ATR period passed to trend classifier (align with execution ``atr_trailing_period`` when possible)."""
    block_entries_session_regime_asia: bool = False
    block_entries_volatility_high: bool = False


class LiveScheduler:
    """
    Wires feed → buffer → features → regime snapshot → signal → risk → paper broker
    (or :class:`broker.gateway.ExecutionGateway` when ``execution_gateway`` is set).

    Call :meth:`step_once` when new data should be ingested (e.g. on a timer).
    """

    def __init__(
        self,
        *,
        feed: Any,
        feature_service: LiveFeatureService,
        signal_service: SignalService,
        risk_gate: RiskGate,
        paper_broker: PaperBroker,
        state_store: LiveStateStore | None = None,
        scheduler_cfg: LiveSchedulerConfig | None = None,
        alert_sink: AlertSink | None = None,
        quarantine: Any = None,
        execution_gateway: Any | None = None,
    ) -> None:
        self._feed = feed
        self._features = feature_service
        self._signal = signal_service
        self._risk = risk_gate
        self._broker = paper_broker
        self._execution_gateway = execution_gateway
        self._store = state_store
        self._sched = scheduler_cfg or LiveSchedulerConfig()
        self._alerts = alert_sink
        self._quarantine = quarantine
        self._buffer: pd.DataFrame = pd.DataFrame()
        self._portfolio_track: PortfolioState | None = None
        self._last_processed_bar_ts: str | None = None
        self._entries_blocked_trend_downtrend: int = 0
        self._entries_blocked_session_asia: int = 0
        self._entries_blocked_high_vol: int = 0
        logger.info(
            "live_execution_assumptions_loaded fill_basis=%s policy=%s costs=%s",
            paper_broker.assumptions.fill_basis,
            paper_broker.assumptions.stop_target_fill_policy,
            paper_broker.assumptions.costs,
        )
        if execution_gateway is not None:
            logger.info(
                "live_execution_gateway_active mode=%s",
                getattr(getattr(execution_gateway, "config", None), "execution_mode", "?"),
            )

    @property
    def buffer(self) -> pd.DataFrame:
        return self._buffer.copy()

    def peek_signal_after_hypothetical_push(
        self,
        incoming: pd.DataFrame,
    ) -> tuple[LiveSignal | None, dict[str, Any] | None, list[str]]:
        """
        Compute the signal that would result if ``incoming`` were merged next, without mutating
        scheduler buffer or executing risk/broker (        Phase 21 opposite-signal handling).
        """
        notes: list[str] = []
        buf, rep = merge_ohlcv_buffer(
            self._buffer,
            incoming,
            expected_freq=self._sched.expected_bar_delta,
            quarantine=self._quarantine,
        )
        if rep.notes:
            notes.extend(rep.notes)
        if len(buf) == 0:
            return None, None, notes + ["peek_empty_buffer"]

        last_row = buf.iloc[-1]
        self._features.set_buffer(buf)
        try:
            try:
                X, feat_row = self._features.latest_model_vector()
            except (ValueError, KeyError) as exc:
                return None, None, notes + [f"peek_features_skipped:{exc}"]

            regime_snap = self._signal.regime_service.snapshot_from_row(feat_row)
            bypass = self._sched.test_only_force_signal_direction is not None
            sig, model_diag = self._signal.generate_with_diagnostics(
                X,
                row=feat_row,
                regime_snapshot=regime_snap,
                ohlcv_buffer=buf,
                strategy_gate_bypass=bypass,
            )
            signal_diagnostics: dict[str, Any] | None = (
                dict(model_diag) if self._sched.signal_diagnostics_enabled else None
            )
            if sig is None and model_diag.get("model_skipped"):
                notes.append(
                    f"strategy_setup_gate:{model_diag.get('strategy_setup_gate', 'unknown')}"
                )

            forced = self._sched.test_only_force_signal_direction
            if sig is not None and forced is not None:
                if signal_diagnostics is None:
                    signal_diagnostics = {}
                signal_diagnostics["test_only_forced_signal"] = True
                signal_diagnostics["test_only_force_label"] = (
                    "DEMO_AUTOMATION_TEST_ONLY_NOT_FOR_PRODUCTION"
                )
                signal_diagnostics["direction_before_test_override"] = int(sig.direction)
                signal_diagnostics["test_override_requested_direction"] = int(forced)
                sig = replace(sig, direction=int(forced))
                signal_diagnostics["direction_after_test_override"] = int(sig.direction)
            elif signal_diagnostics is not None:
                signal_diagnostics["test_only_forced_signal"] = False

            if signal_diagnostics is not None and sig is not None:
                signal_diagnostics["final_direction"] = int(sig.direction)

            return sig, signal_diagnostics, notes
        finally:
            self._features.set_buffer(self._buffer)

    def load_persisted(self, state: LivePersistedState) -> None:
        """Hydrate buffer tail, broker, and portfolio trackers."""
        self._buffer = LiveStateStore.ohlcv_from_records(state.ohlcv_tail_records)
        self._features.set_buffer(self._buffer)
        if state.paper:
            paper_broker_apply_dict(self._broker, state.paper)
        if state.portfolio:
            self._portfolio_track = PortfolioState.from_dict(state.portfolio)
        self._last_processed_bar_ts = state.last_processed_timestamp
        logger.info(
            "live_state_restored last_processed_timestamp=%s restart_detected=True",
            self._last_processed_bar_ts,
        )

    def build_portfolio_state(
        self,
        bar_timestamp: pd.Timestamp,
        last_close: float,
        session_tag: str,
    ) -> PortfolioState:
        snap = self._broker.portfolio_snapshot(last_close)
        ps = PortfolioState(
            equity=float(snap["equity"]),
            cash=float(snap["cash"]),
            unrealized_pnl=float(snap["unrealized_pnl"]),
            realized_pnl=float(snap["realized_pnl"]),
            open_position_count=int(snap["open_position_count"]),
            last_price=float(last_close),
            contract_size=float(self._broker.config.contract_size),
            net_exposure_shares=float(snap["net_exposure_shares"]),
        )
        if self._portfolio_track is not None:
            ps.peak_equity_today = self._portfolio_track.peak_equity_today
            ps.day_anchor = self._portfolio_track.day_anchor
            ps.session_peak_equity = dict(self._portfolio_track.session_peak_equity)
            ps.last_session_tag = self._portfolio_track.last_session_tag
        ps.on_equity_tick(bar_timestamp, ps.equity, session_tag)
        self._portfolio_track = ps
        return ps

    def step_once(
        self,
        *,
        quote: QuoteSnapshot | None = None,
    ) -> SchedulerStepResult:
        notes: list[str] = []
        try:
            incoming = self._feed.poll()
            self._buffer, rep = merge_ohlcv_buffer(
                self._buffer,
                incoming,
                expected_freq=self._sched.expected_bar_delta,
                quarantine=self._quarantine,
            )
            if rep.notes:
                notes.extend(rep.notes)
            if rep.n_quarantined:
                notes.append(f"quarantined_incoming_rows={rep.n_quarantined}")
                logger.warning("live_quarantine merged_batch quarantined=%s", rep.n_quarantined)
            self._features.set_buffer(self._buffer)

            if len(self._buffer) == 0:
                return SchedulerStepResult(
                    timestamp=None,
                    signal=None,
                    risk=None,
                    executed=False,
                    notes=["empty_buffer"],
                )

            last_row = self._buffer.iloc[-1]
            ts = pd.Timestamp(last_row["timestamp"])
            o, h, l, c = (
                float(last_row["open"]),
                float(last_row["high"]),
                float(last_row["low"]),
                float(last_row["close"]),
            )

            try:
                X, feat_row = self._features.latest_model_vector()
            except (ValueError, KeyError) as exc:
                live_alert(
                    self._alerts,
                    "runtime_error",
                    f"feature_warmup_or_schema: {exc}",
                    exception_type=type(exc).__name__,
                )
                return SchedulerStepResult(
                    timestamp=ts,
                    signal=None,
                    risk=None,
                    executed=False,
                    notes=notes + [f"features_skipped:{exc}"],
                )

            regime_snap = self._signal.regime_service.snapshot_from_row(feat_row)
            bypass = self._sched.test_only_force_signal_direction is not None
            sig, model_diag = self._signal.generate_with_diagnostics(
                X,
                row=feat_row,
                regime_snapshot=regime_snap,
                ohlcv_buffer=self._buffer,
                strategy_gate_bypass=bypass,
            )
            signal_diagnostics: dict[str, Any] | None = (
                dict(model_diag) if self._sched.signal_diagnostics_enabled else None
            )
            if sig is None and model_diag.get("model_skipped"):
                notes.append(
                    f"strategy_setup_gate:{model_diag.get('strategy_setup_gate', 'unknown')}"
                )

            forced = self._sched.test_only_force_signal_direction
            if sig is not None and forced is not None:
                if signal_diagnostics is None:
                    signal_diagnostics = {}
                signal_diagnostics["test_only_forced_signal"] = True
                signal_diagnostics["test_only_force_label"] = (
                    "DEMO_AUTOMATION_TEST_ONLY_NOT_FOR_PRODUCTION"
                )
                signal_diagnostics["direction_before_test_override"] = int(sig.direction)
                signal_diagnostics["test_override_requested_direction"] = int(forced)
                sig = replace(sig, direction=int(forced))
                signal_diagnostics["direction_after_test_override"] = int(sig.direction)
            elif signal_diagnostics is not None:
                signal_diagnostics["test_only_forced_signal"] = False

            if signal_diagnostics is not None and sig is not None:
                signal_diagnostics["final_direction"] = int(sig.direction)

            ts_key = str(ts.isoformat())
            if (
                self._sched.enable_bar_dedupe
                and self._last_processed_bar_ts is not None
                and ts_key == self._last_processed_bar_ts
            ):
                logger.info("live_skipped_replayed_bar timestamp=%s (no duplicate trades)", ts_key)
                return SchedulerStepResult(
                    timestamp=ts,
                    signal=sig,
                    risk=None,
                    executed=False,
                    notes=notes + ["skipped_replayed_bar"],
                    signal_diagnostics=signal_diagnostics,
                )

            if sig is None:
                self.build_portfolio_state(ts, c, regime_snap.session)
                if self._store is not None:
                    self._persist(None, ts)
                self._last_processed_bar_ts = ts_key
                return SchedulerStepResult(
                    timestamp=ts,
                    signal=None,
                    risk=None,
                    executed=False,
                    notes=notes,
                    signal_diagnostics=signal_diagnostics,
                )

            live_alert(
                self._alerts,
                "signal_generated",
                "inference",
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=sig.confidence,
                timestamp=str(sig.timestamp),
            )

            port = self.build_portfolio_state(ts, c, sig.regime.session)
            risk = self._risk.evaluate(RiskContext(signal=sig, portfolio=port, quote=quote))

            executed = False
            ex_dec: Any = None
            if risk.approved:
                regime_entry_block = False
                if int(sig.direction) != 0 and int(port.open_position_count) == 0:
                    unsupported = []
                    if self._sched.block_entries_trend_regime_downtrend:
                        unsupported.append("trend")
                    if self._sched.block_entries_session_regime_asia:
                        unsupported.append("session")
                    if self._sched.block_entries_volatility_high:
                        unsupported.append("volatility")
                    if unsupported:
                        # Removed classifiers must never silently claim an entry decision.
                        regime_entry_block = True
                        notes.append("unsupported_regime_entry_gate:" + ",".join(unsupported))

                fills: list[Any] = []
                if not regime_entry_block:
                    if self._execution_gateway is not None:
                        from broker.models import OrderRequest
                        from broker.validators import direction_to_side, quantity_for_direction

                        req = OrderRequest(
                            symbol=sig.symbol,
                            side=direction_to_side(sig.direction),
                            quantity=quantity_for_direction(sig.direction),
                            order_type="market",
                            timestamp=ts,
                            execution_mode=self._execution_gateway.mode.value,
                            target_direction=int(sig.direction),
                            bar_open=o, bar_high=h, bar_low=l, bar_close=c,
                            run_id=self._sched.run_id,
                            signal_confidence=sig.confidence,
                            primary_class=sig.primary_class,
                        )
                        ex_dec = self._execution_gateway.process_risk_approved_bar(
                            req, portfolio=port
                        )
                        executed = bool(ex_dec.success)
                    else:
                        fills = self._broker.step(ts, o, h, l, c, sig.direction)
                        executed = len(fills) > 0
                for fl in fills:
                    if fl.reason == "open":
                        live_alert(
                            self._alerts,
                            "trade_opened",
                            fl.reason,
                            side=fl.side,
                            qty=fl.quantity,
                            price=fl.price,
                        )
                    elif fl.reason in ("stop_loss", "take_profit", "signal_rebalance"):
                        live_alert(
                            self._alerts,
                            "trade_closed",
                            fl.reason,
                            side=fl.side,
                            qty=fl.quantity,
                            price=fl.price,
                        )
            else:
                live_alert(
                    self._alerts,
                    "risk_blocked",
                    "risk_gate",
                    reasons=risk.reasons,
                    direction=sig.direction,
                )

            self.build_portfolio_state(ts, c, sig.regime.session)

            if self._store is not None:
                self._persist(sig, ts)
            self._last_processed_bar_ts = ts_key

            return SchedulerStepResult(
                timestamp=ts,
                signal=sig,
                risk=risk,
                executed=executed,
                notes=notes,
                execution_decision=ex_dec,
                signal_diagnostics=signal_diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("live scheduler step failed")
            live_alert(
                self._alerts,
                "runtime_error",
                f"step_once: {exc}",
                exception_type=type(exc).__name__,
            )
            raise

    def _persist(self, sig: LiveSignal | None, ts: pd.Timestamp) -> None:
        assert self._store is not None
        port = self._portfolio_track
        st = LivePersistedState(
            last_processed_timestamp=str(ts.isoformat()),
            ohlcv_tail_records=LiveStateStore.ohlcv_to_records(
                self._buffer,
                self._sched.persist_tail_bars,
            ),
            max_tail_bars=self._sched.persist_tail_bars,
            paper=paper_broker_to_dict(self._broker),
            portfolio=port.to_dict() if port is not None else {},
            latest_signal=live_signal_to_dict(sig) if sig is not None else {},
            model_primary_path=self._sched.model_primary_path,
            model_meta_path=self._sched.model_meta_path,
            feature_columns=list(self._sched.feature_columns or []),
        )
        self._store.save(st)
