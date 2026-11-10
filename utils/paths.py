"""Convenience re-exports for path resolution used across packages."""

from __future__ import annotations

from pathlib import Path

from config.settings import ProjectPaths, get_project_paths

__all__ = ["ProjectPaths", "get_project_paths", "root", "confined_output_subdir"]

# Convenience for smoke tests / docs
root = get_project_paths().root


def confined_output_subdir(output_root: Path, *parts: str, run_id: str) -> Path:
    """Resolve ``output_root/parts.../run_id`` and reject absolute/traversal escapes.

    ``run_id`` must be a single relative path component (no separators, ``.``, or ``..``).
    """
    candidate_id = Path(run_id)
    if candidate_id.is_absolute() or len(candidate_id.parts) != 1 or candidate_id.name in {"", ".", ".."}:
        raise ValueError("run_id must be a single relative path component")
    root_resolved = output_root.resolve()
    candidate = root_resolved.joinpath(*parts, candidate_id).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError("resolved artifact directory escapes project outputs/")
    return candidate
