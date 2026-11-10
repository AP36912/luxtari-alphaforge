"""
Broker connection health: latency, last success, safe reconnect with backoff (Phase 18).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from broker.base import BrokerAdapter

logger = logging.getLogger("quant_ai.broker.connection_health")


@dataclass
class HealthReport:
    """Structured health snapshot for logs and JSON reports."""

    connected: bool
    ok: bool
    message: str
    adapter_name: str
    latency_ms: float | None
    last_success_utc: str | None
    reconnect_attempts: int = 0
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("extra") is None:
            d.pop("extra", None)
        return d


class BrokerConnectionHealthMonitor:
    """
    Wraps a :class:`BrokerAdapter` with periodic checks and bounded exponential backoff reconnect.
    """

    def __init__(
        self,
        adapter: BrokerAdapter,
        *,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 60.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self._adapter = adapter
        self._backoff_initial = backoff_initial_s
        self._backoff_max = backoff_max_s
        self._backoff_mult = backoff_multiplier
        self._failures = 0
        self._last_success_utc: str | None = None
        self._reconnect_attempts = 0

    @property
    def last_success_utc(self) -> str | None:
        return self._last_success_utc

    def check_once(self) -> HealthReport:
        """Single health probe (may call adapter APIs)."""
        t0 = time.perf_counter()
        ok, msg = self._adapter.health_check()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if ok:
            self._last_success_utc = datetime.now(timezone.utc).isoformat()
            self._failures = 0
        report = HealthReport(
            connected=ok and bool(getattr(self._adapter, "_connected", True)),
            ok=ok,
            message=msg,
            adapter_name=getattr(self._adapter, "name", type(self._adapter).__name__),
            latency_ms=round(latency_ms, 3),
            last_success_utc=self._last_success_utc,
            reconnect_attempts=self._reconnect_attempts,
        )
        logger.info(
            "broker_health ok=%s adapter=%s latency_ms=%s msg=%s",
            ok,
            report.adapter_name,
            report.latency_ms,
            msg,
        )
        return report

    def reconnect_with_backoff(self, *, max_attempts: int = 5) -> HealthReport:
        """
        Disconnect, sleep with jittered exponential backoff, then connect again.
        Returns the post-reconnect :class:`HealthReport` from :meth:`check_once`.
        """
        attempt = 0
        delay = self._backoff_initial
        self._adapter.disconnect()
        while attempt < max_attempts:
            attempt += 1
            self._reconnect_attempts += 1
            jitter = random.uniform(0, 0.25 * delay)
            sleep_s = min(self._backoff_max, delay) + jitter
            logger.warning(
                "broker_reconnect_attempt n=%s sleep_s=%.2f failures=%s",
                attempt,
                sleep_s,
                self._failures,
            )
            time.sleep(sleep_s)
            self._adapter.connect()
            rep = self.check_once()
            if rep.ok:
                return replace(rep, reconnect_attempts=self._reconnect_attempts)
            delay *= self._backoff_mult
            self._failures += 1
        return self.check_once()
