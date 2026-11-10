"""Deep-merge template fragments into experiment dicts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, TYPE_CHECKING

from research_templates.base import PUBLIC_RESEARCH_BASE, SYNTH_RESEARCH_BASE, deep_merge
from research_templates.registry import get_template_fragment, list_template_ids

if TYPE_CHECKING:
    from config.experiment_schema import ExperimentConfig


def _strip_template_directives(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k not in ("template", "extends_template")}


def apply_named_template(template_id: str, overlay: dict[str, Any]) -> dict[str, Any]:
    overlay = _strip_template_directives(overlay)
    body = deep_merge(copy.deepcopy(SYNTH_RESEARCH_BASE), get_template_fragment(template_id))
    return deep_merge(body, overlay)


def experiment_dict_from_template(
    template_id: str,
    overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return apply_named_template(template_id, overlay or {})


def build_experiment_config_from_template(
    template_id: str,
    *,
    overlay: dict[str, Any] | None = None,
    source_path: Path | None = None,
) -> "ExperimentConfig":
    from config.experiment_schema import experiment_config_from_dict

    d = experiment_dict_from_template(template_id, overlay or {})
    return experiment_config_from_dict(d, source_path=source_path)


def resolve_unknown_template(template_id: str) -> str:
    if template_id in list_template_ids():
        return template_id
    known = ", ".join(list_template_ids())
    raise KeyError(f"Unknown research template {template_id!r}. Known: {known}")
