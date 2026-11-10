"""Public execution surface fails closed for removed broker capabilities."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from broker.gateway import ExecutionGateway, GatewayConfig
from broker.mode import normalize_execution_mode
from broker.models import CancelRequest, OrderRequest, PositionCloseRequest
from config.experiment_schema import ExecutionConfig, execution_config_from_dict, experiment_config_from_dict
from config.pipeline import _confined_run_dir


def _req(mode="paper", ts=None, client_id=None):
    return OrderRequest(symbol="SYNTH", side="buy", quantity=1.0, order_type="market",
                        timestamp=ts or pd.Timestamp(datetime.now(timezone.utc)), execution_mode=mode,
                        target_direction=1, bar_open=100., bar_high=101., bar_low=99., bar_close=100.,
                        client_order_id=client_id)


@pytest.mark.parametrize("mode", ["broker_live", "demo", "broker_demo", "mt5", "ibkr", ""])
def test_forbidden_modes(mode):
    with pytest.raises(ValueError): normalize_execution_mode(mode)


@pytest.mark.parametrize("key", ["broker_live_enabled", "broker_submit_enabled", "removed_mt5_login",
                                  "mt5_server", "demo_lifecycle_max_poll_attempts", "automation_enabled",
                                  "dry_run_shadow_mode"])
def test_forbidden_execution_keys(key):
    with pytest.raises(ValueError, match="forbidden"): execution_config_from_dict({key: True})


def test_public_config_has_no_live_fields():
    names = {f.name for f in fields(ExecutionConfig)}
    assert not any(n.startswith(("removed_", "mt5", "demo_lifecycle", "automation_")) for n in names)
    assert "broker_live_enabled" not in names and "broker_submit_enabled" not in names


def test_legacy_live_enablement_rejected():
    with pytest.raises(ValueError, match="legacy live"): experiment_config_from_dict({"live": {"broker_live_enabled": True}})


def test_emergency_and_kill_file_reject(tmp_path: Path):
    emergency = ExecutionGateway(GatewayConfig(emergency_stop=True))
    assert not emergency.process_risk_approved_bar(_req()).success
    kill = tmp_path / "KILL"; kill.write_text("stop")
    file_gateway = ExecutionGateway(GatewayConfig(kill_switch_file_path=str(kill)))
    assert not file_gateway.process_risk_approved_bar(_req()).success


def test_size_risk_rejection_and_monitoring_flag_not_halt():
    limited = ExecutionGateway(GatewayConfig(max_position_size=0.5, stale_signal_max_age_seconds=None))
    result = limited.process_risk_approved_bar(_req())
    assert not result.success and "max_position_size_exceeded" in (result.rejection_reason or "")

    from broker.safety import SafetyState, evaluate_execution_safety
    from live.portfolio_state import PortfolioState
    cfg = ExecutionConfig(kill_switch_enabled=True, stale_signal_max_age_seconds=None)
    portfolio = PortfolioState(equity=100_000, cash=100_000, unrealized_pnl=0, realized_pnl=0,
                               open_position_count=0, last_price=100)
    evaluation = evaluate_execution_safety(_req(), cfg, SafetyState(), portfolio)
    assert evaluation.allowed


def test_stale_and_duplicate_rejection():
    stale = ExecutionGateway(GatewayConfig(stale_signal_max_age_seconds=1))
    old = pd.Timestamp("2020-01-01", tz="UTC")
    assert "stale_signal" in (stale.process_risk_approved_bar(_req(ts=old)).rejection_reason or "")
    duplicate = ExecutionGateway(GatewayConfig(stale_signal_max_age_seconds=None, duplicate_order_cooldown_seconds=60))
    assert duplicate.process_risk_approved_bar(_req(client_id="same")).success
    result = duplicate.process_risk_approved_bar(_req(client_id="same"))
    assert not result.success and "duplicate_order_cooldown" in (result.rejection_reason or "")


def test_no_mt5_broker_module_exists():
    broker_dir = Path(__file__).parents[1] / "broker"
    assert not any("mt5" in p.name.lower() for p in broker_dir.glob("*.py"))


def test_unknown_top_level_config_key_rejected():
    with pytest.raises(ValueError, match="unknown top-level"):
        experiment_config_from_dict({"unexpected": {}})


@pytest.mark.parametrize("section", ["live", "live_ops", "execution"])
@pytest.mark.parametrize("key", ["mt5_server", "bridge_url", "broker_live_mode", "broker_submit_flag", "credentials"])
def test_forbidden_broker_keys_rejected_recursively(section, key):
    with pytest.raises(ValueError, match="forbidden"):
        experiment_config_from_dict({section: {"nested": {key: False}}})


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/escape", "nested/escape"])
def test_run_id_cannot_escape_outputs(tmp_path: Path, run_id: str):
    with pytest.raises(ValueError, match="run_id|escapes"):
        _confined_run_dir(tmp_path / "outputs", run_id)


def test_kill_switch_read_error_blocks(tmp_path: Path):
    import os
    from broker.safety import SafetyState, evaluate_execution_safety, _kill_switch_file_status
    from live.portfolio_state import PortfolioState
    # Owner can still stat mode-000 files; block directory execute to force PermissionError.
    blocked_dir = tmp_path / "noenter"
    blocked_dir.mkdir()
    kill = blocked_dir / "KILL"
    kill.write_text("stop", encoding="utf-8")
    os.chmod(blocked_dir, 0)
    try:
        assert _kill_switch_file_status(kill) == (False, True)
        cfg = ExecutionConfig(stale_signal_max_age_seconds=None)
        portfolio = PortfolioState(equity=100_000, cash=100_000, unrealized_pnl=0, realized_pnl=0,
                                   open_position_count=0, last_price=100)
        result = evaluate_execution_safety(_req(), cfg, SafetyState(), portfolio,
                                           kill_switch_file_path=kill)
        assert not result.allowed
        assert "kill_switch_file_read_error" in result.reasons
    finally:
        os.chmod(blocked_dir, 0o755)


def test_metadata_forbidden_credentials_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        experiment_config_from_dict({"metadata": {"credentials": "x"}})


@pytest.mark.parametrize("section,key", [
    ("data", "unexpected_field"),
    ("validation", "unexpected_field"),
    ("output", "unexpected_field"),
    ("audit", "unexpected_field"),
    ("tracking", "unexpected_field"),
    ("live_ops", "unexpected_field"),
])
def test_unknown_nested_structured_keys_rejected(section, key):
    with pytest.raises(ValueError, match="unknown"):
        experiment_config_from_dict({section: {key: True}})


def test_bundle_run_id_cannot_escape(tmp_path: Path):
    from reporting.experiment_bundle import write_experiment_bundle
    with pytest.raises(ValueError, match="run_id|escapes"):
        write_experiment_bundle(
            run_id="../escape",
            run_dir=tmp_path,
            reports_dir=tmp_path,
            metrics_payload={},
            audit_findings=[],
            feature_list=[],
            label_distribution={},
            model_parameters={},
            config_snapshot={},
            metadata={},
            label_spec_dict={},
            model_name="x",
            resolved_data_path="x",
            outputs_dir=tmp_path / "outputs",
        )


def test_demo_cancel_and_close_are_explicitly_unsupported():
    from broker.demo_adapter import DemoBrokerAdapter
    adapter = DemoBrokerAdapter()
    cancel = adapter.cancel_order(CancelRequest(symbol="SYNTH", broker_order_id="demo"))
    close = adapter.close_position(PositionCloseRequest(symbol="SYNTH"))
    assert not cancel.success and cancel.rejection_reason == "not_supported"
    assert not close.success and close.rejection_reason == "not_supported"


def test_secret_aliases_rejected():
    for key in ("token", "client_secret", "access_token", "automation_enabled"):
        with pytest.raises(ValueError, match="forbidden"):
            experiment_config_from_dict({"metadata": {key: "x"}})


def test_unknown_regime_key_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        experiment_config_from_dict({"regime": {"not_a_real_field": True}})


def test_missing_kill_switch_file_allows_when_not_present(tmp_path: Path):
    from broker.safety import SafetyState, evaluate_execution_safety, _kill_switch_file_status
    assert _kill_switch_file_status(tmp_path / "absent") == (False, False)
    from live.portfolio_state import PortfolioState
    cfg = ExecutionConfig(stale_signal_max_age_seconds=None)
    portfolio = PortfolioState(equity=100_000, cash=100_000, unrealized_pnl=0, realized_pnl=0,
                               open_position_count=0, last_price=100)
    result = evaluate_execution_safety(_req(), cfg, SafetyState(), portfolio,
                                       kill_switch_file_path=tmp_path / "absent")
    assert result.allowed
