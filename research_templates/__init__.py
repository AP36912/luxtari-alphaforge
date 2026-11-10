"""Public research templates."""

from research_templates.base import PUBLIC_RESEARCH_BASE, SYNTH_RESEARCH_BASE
from research_templates.registry import get_template, list_template_ids, list_templates

__all__ = [
    "PUBLIC_RESEARCH_BASE",
    "SYNTH_RESEARCH_BASE",
    "get_template",
    "list_templates",
    "list_template_ids",
]
