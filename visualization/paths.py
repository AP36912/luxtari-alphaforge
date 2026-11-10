"""Default report output directory (under project root)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """``quant_ai/`` root (parent of ``visualization/``)."""
    return Path(__file__).resolve().parents[1]


REPORTS_DIR = project_root() / "outputs" / "reports"


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR
