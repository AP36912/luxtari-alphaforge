"""
Write invalid live bars to disk with metadata (Phase 16.5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from live.bar_validation import BarValidationResult, utc_now_iso

logger = logging.getLogger("quant_ai.live.quarantine")


class QuarantineWriter:
    """Append JSON lines under ``base_dir`` (e.g. ``outputs/live/quarantine``)."""

    def __init__(self, base_dir: str | Path, *, symbol: str = "", enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.symbol = symbol
        self.enabled = enabled
        self._count = 0

    @property
    def incident_count(self) -> int:
        return self._count

    def write_invalid(
        self,
        raw_row: dict[str, Any],
        validation: BarValidationResult,
        *,
        context: str = "merge",
    ) -> Path | None:
        if not self.enabled:
            return None
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / "quarantine.jsonl"
        rec = {
            "ingestion_time_utc": utc_now_iso(),
            "symbol": self.symbol or None,
            "context": context,
            "validation_reasons": list(validation.reasons),
            "raw_row": {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in raw_row.items()},
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self._count += 1
        logger.warning(
            "quarantined_invalid_bar",
            extra={
                "symbol": self.symbol,
                "reasons": validation.reasons,
                "path": str(path),
            },
        )
        return path
