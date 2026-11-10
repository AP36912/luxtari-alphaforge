"""Publication hygiene checks for the public release tree."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".pytest_cache", "outputs", "__pycache__"}
_SKIP_NAMES = {"SHA256SUMS", "EXPORT_MANIFEST.txt", "EXPORT_FILE_INVENTORY.txt"}
# Generic absolute home-directory patterns (no operator-specific username literal).
_LEAD = r"(^|[\s\"\'`=(])"
_HOME_PATTERNS = (
    re.compile(_LEAD + r"/home/[A-Za-z0-9._-]+", re.M),
    re.compile(_LEAD + r"/Users/[A-Za-z0-9._-]+", re.M),
    re.compile(r"(?i)C:\\Users\\[A-Za-z0-9._-]+"),
)


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in _SKIP_NAMES:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".npz", ".pt", ".joblib"}:
            continue
        yield path


def test_no_absolute_home_paths_in_tree() -> None:
    hits = []
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8", errors="strict")
        if any(p.search(text) for p in _HOME_PATTERNS):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], f"absolute home paths remain in: {hits}"


def test_removed_placeholders_absent_from_public_python() -> None:
    hits = []
    for path in (ROOT / "evaluation").rglob("*.py"):
        if "REMOVED_" in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(ROOT)))
    for path in (ROOT / "live").rglob("*.py"):
        if "REMOVED_" in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], f"REMOVED_ residue remains in: {hits}"
