"""Public gateway/mode fail-closed tests."""
from broker.mode import ExecutionMode, normalize_execution_mode
from broker.gateway import ExecutionGateway, GatewayConfig
import pytest


def test_allowed_modes():
    assert normalize_execution_mode("paper") is ExecutionMode.PAPER
    assert normalize_execution_mode("sim") is ExecutionMode.SIM
    assert normalize_execution_mode("dry_run") is ExecutionMode.DRY_RUN


def test_rejects_live_mode():
    with pytest.raises(ValueError):
        normalize_execution_mode("broker_live")


def test_gateway_config_excludes_submit_flag():
    with pytest.raises(TypeError):
        GatewayConfig(broker_submit_enabled=True)
