"""Reproducibility helpers."""

from __future__ import annotations

from audit.repro import assert_numpy_determinism_after_seed


def test_numpy_determinism_after_seed() -> None:
    assert assert_numpy_determinism_after_seed(12345, trials=3) is True
