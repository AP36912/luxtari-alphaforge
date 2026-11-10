"""Load experiment YAML into :class:`ExperimentConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.experiment_schema import ExperimentConfig, experiment_config_from_dict
from research_templates.merge import deep_merge


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment YAML root must be a mapping: {path}")
    return raw


def _resolve_extends(path: Path, *, _chain: list[Path] | None = None) -> dict[str, Any]:
    """
    If the file defines ``extends_config`` (or legacy ``extends``), load that
    file first and deep-merge this file on top (child wins). Paths are relative
    to the including file's directory unless absolute.
    """
    _chain = _chain or []
    path = path.resolve()
    if path in _chain:
        raise ValueError(f"extends_config cycle: {' -> '.join(str(x) for x in _chain + [path])}")
    raw = _load_yaml_file(path)
    ext = raw.get("extends_config") or raw.get("extends")
    if not ext:
        return dict(raw)
    ext_path = Path(ext)
    if not ext_path.is_absolute():
        ext_path = (path.parent / ext_path).resolve()
    if not ext_path.is_file():
        raise FileNotFoundError(f"extends_config target not found: {ext_path} (from {path})")
    base = _resolve_extends(ext_path, _chain=_chain + [path])
    overlay = {k: v for k, v in raw.items() if k not in ("extends_config", "extends", "template", "extends_template")}
    return deep_merge(base, overlay)


def load_experiment_yaml(path: str | Path) -> ExperimentConfig:
    """
    Load YAML. If the root mapping contains ``template`` or ``extends_template``,
    merge the named :mod:`research_templates` preset first, then apply remaining
    keys (so YAML overrides the template).

    If the file contains ``extends_config`` / ``extends``, that base YAML is
    loaded first and the current file is merged on top (see :func:`_resolve_extends`).
    """
    p = Path(path).resolve()
    raw: dict[str, Any] = _resolve_extends(p)
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML root must be a mapping")
    tpl = raw.get("template") or raw.get("extends_template")
    if tpl is not None:
        from research_templates.merge import apply_named_template

        if not isinstance(tpl, str) or not tpl.strip():
            raise ValueError("template / extends_template must be a non-empty string")
        raw = apply_named_template(tpl.strip(), raw)
    return experiment_config_from_dict(raw, source_path=p)
