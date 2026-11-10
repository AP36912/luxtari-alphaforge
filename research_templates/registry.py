"""Template registry — public minimal templates only."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from research_templates.base import PUBLIC_RESEARCH_BASE, deep_merge


_FRAGMENTS: dict[str, dict[str, Any]] = {
    "minimal_rf_fixture": {
        "metadata": {"name": "minimal_rf_fixture"},
        "model": {"name": "random_forest"},
        "data": {"path": "tests/fixtures/sample_ohlcv_experiment.csv"},
    },
    "minimal_public_experiment": {
        "metadata": {"name": "minimal_public_experiment"},
        "model": {"name": "random_forest"},
        "data": {"path": "tests/fixtures/sample_ohlcv_experiment.csv"},
    },
}


def list_template_ids() -> list[str]:
    return sorted(_FRAGMENTS)


def list_templates() -> list[str]:
    return list_template_ids()


def get_template_fragment(name: str) -> dict[str, Any]:
    if name not in _FRAGMENTS:
        known = ", ".join(list_template_ids())
        raise KeyError(f"Unknown research template {name!r}. Known: {known}")
    return deepcopy(_FRAGMENTS[name])


def get_template(name: str) -> dict[str, Any]:
    return deep_merge(deepcopy(PUBLIC_RESEARCH_BASE), get_template_fragment(name))
