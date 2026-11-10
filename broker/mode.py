"""Execution mode helpers — public candidate (paper/sim only)."""

from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    PAPER = "paper"
    DRY_RUN = "dry_run"
    SIM = "sim"


ALLOWED_MODES = {m.value for m in ExecutionMode}


def normalize_execution_mode(raw: str | None) -> ExecutionMode:
    value = (ExecutionMode.PAPER.value if raw is None else str(raw)).strip().lower()
    if value not in ALLOWED_MODES:
        raise ValueError(
            f"unsupported execution_mode={raw!r}; allowed={sorted(ALLOWED_MODES)}"
        )
    return ExecutionMode(value)


parse_execution_mode = normalize_execution_mode


def assert_public_execution_mode(mode: ExecutionMode) -> None:
    if mode not in {ExecutionMode.PAPER, ExecutionMode.DRY_RUN, ExecutionMode.SIM}:
        raise ValueError(f"public edition refuses mode={mode}")
