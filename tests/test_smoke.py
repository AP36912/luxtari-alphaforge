"""Smoke tests for public project layout and core capabilities."""
from __future__ import annotations

import utils.paths as paths


def test_project_root_has_required_release_files():
    root = paths.root
    required = (
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "run_experiment.py",
        "requirements-stage3.lock.txt",
        "pytest.ini",
    )
    missing = [name for name in required if not (root / name).is_file()]
    assert missing == [], f"missing required release files: {missing}"


def test_core_public_modules_importable():
    from broker.mode import ALLOWED_MODES, normalize_execution_mode
    from config.experiment_schema import ExperimentConfig
    from research.public_offline_run import run_public_synthetic

    assert normalize_execution_mode("paper").value == "paper"
    assert ALLOWED_MODES == {"paper", "dry_run", "sim"}
    assert ExperimentConfig is not None
    assert callable(run_public_synthetic)


def test_fixture_exists():
    fixture = paths.root / "tests" / "fixtures" / "sample_ohlcv_experiment.csv"
    assert fixture.is_file()
    header = fixture.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("timestamp,open,high,low,close")
