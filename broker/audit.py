"""
Durable execution audit trail (JSONL + optional markdown summaries) — Phase 17.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("quant_ai.broker.audit")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_audit_event(
    record: dict[str, Any],
    *,
    jsonl_path: Path,
) -> str:
    """Append one JSON line; return record id."""
    rid = str(record.get("id") or uuid.uuid4().hex)
    record = {**record, "id": rid, "logged_at_utc": datetime.now(timezone.utc).isoformat()}
    _ensure_parent(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.info("execution_audit_event id=%s event=%s", rid, record.get("event"))
    return rid


def write_markdown_summary(
    lines: list[str],
    *,
    out_dir: Path,
    stem: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
