"""
Phase 20.5 — explain how :class:`~live.live_types.LiveSignal` direction is derived from model output.
"""

from __future__ import annotations

from typing import Any


def classify_direction_mapping(
    primary_class: int,
    class_to_direction: dict[int, int] | None,
) -> str:
    """Label which branch of :func:`live.signal_service._default_direction_from_class` applies."""
    if class_to_direction is not None:
        return "explicit_class_to_direction"
    cls = int(primary_class)
    if cls in (-1, 0, 1):
        return "triple_barrier_identity"
    return "generic_classifier_posneg"


def neutral_flat_explanation(
    *,
    direction: int,
    primary_class: int,
    mapping_kind: str,
    class_to_direction: dict[int, int] | None,
) -> str | None:
    """Human-readable reason when final mapped direction is 0 (flat / no trade side)."""
    if int(direction) != 0:
        return None
    pred = int(primary_class)
    if mapping_kind == "triple_barrier_identity":
        return (
            f"primary_class={pred} with triple_barrier_identity mapping: "
            "direction equals the class label; 0 means hold/side-neutral in {-1,0,+1} space."
        )
    if mapping_kind == "generic_classifier_posneg":
        return (
            f"primary_class={pred} with generic_classifier_posneg mapping: "
            "direction is +1 only when class>0; otherwise 0 (flat)."
        )
    if mapping_kind == "explicit_class_to_direction" and class_to_direction is not None:
        mapped = int(class_to_direction.get(pred, 0))
        return (
            f"primary_class={pred} maps via class_to_direction to {mapped}; "
            "flat means the configured map yields 0 for this class."
        )
    return "direction is 0 (flat) after mapping."
