"""
Central settings loader.

Keeps secrets out of code: use environment variables or .env (not committed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem roots for data and outputs."""

    root: Path
    data_dir: Path
    outputs_dir: Path


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """
    Return canonical project paths relative to the quant_ai repository root.

    Args:
        root: Optional explicit root; defaults to parent of this package's parent.
    """
    if root is None:
        root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    return ProjectPaths(root=root, data_dir=data_dir, outputs_dir=outputs_dir)
