"""
Pluggable alerts for signals, paper fills, risk blocks, and errors.

Default sinks: stderr logging and optional JSON-lines file.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LiveEvent = Literal[
    "signal_generated",
    "trade_opened",
    "trade_closed",
    "risk_blocked",
    "runtime_error",
]


class AlertSink(ABC):
    """Extension point for Slack, email, Prometheus, etc."""

    @abstractmethod
    def emit(self, level: int, event: LiveEvent, message: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError


@dataclass
class LoggingAlertSink(AlertSink):
    """Structured records via :mod:`logging` (console by default)."""

    logger_name: str = "quant_ai.live"
    level: int = logging.INFO

    def __post_init__(self) -> None:
        self._log = logging.getLogger(self.logger_name)

    def emit(self, level: int, event: LiveEvent, message: str, payload: dict[str, Any]) -> None:
        rec = {"event": event, "message": message, **payload}
        self._log.log(level, "%s", json.dumps(rec, default=str))


@dataclass
class FileJsonlSink(AlertSink):
    """Append one JSON object per line (easy to ship to observability later)."""

    path: Path

    def emit(self, level: int, event: LiveEvent, message: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"level": level, "event": event, "message": message, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class MultiSink(AlertSink):
    def __init__(self, sinks: list[AlertSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, level: int, event: LiveEvent, message: str, payload: dict[str, Any]) -> None:
        for s in self._sinks:
            s.emit(level, event, message, payload)


def live_alert(
    sink: AlertSink | None,
    event: LiveEvent,
    message: str,
    *,
    level: int = logging.INFO,
    **payload: Any,
) -> None:
    if sink is None:
        return
    sink.emit(level, event, message, payload)
