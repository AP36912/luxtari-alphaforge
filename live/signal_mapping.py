"""
Phase 21 — configurable label→direction mapping and probability gates (demo / live inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from live.signal_diagnostics import classify_direction_mapping, neutral_flat_explanation


def default_direction_from_class(cls: int, class_to_direction: dict[int, int] | None) -> int:
    """Map primary predicted class to {-1,0,+1} (same rules as pre-Phase-21 ``SignalService``)."""
    if class_to_direction is None:
        if int(cls) in (-1, 0, 1):
            return int(cls)
        return 1 if int(cls) > 0 else 0
    return int(class_to_direction.get(int(cls), 0))


def merge_class_direction_maps(
    research_map: dict[int, int] | None,
    override: dict[int, int] | None,
) -> dict[int, int] | None:
    """
    Merge research ``class_to_direction`` with execution ``signal_class_mapping_override``.

    Override keys replace; remaining keys fall back to research_map.
    Returns None if both are None/empty (legacy default mapping).
    """
    if override is None or len(override) == 0:
        return research_map
    base = dict(research_map or {})
    for k, v in override.items():
        base[int(k)] = int(v)
    return base


def per_class_probabilities(
    classes_list: list[Any] | None,
    raw_proba_row: list[float] | None,
) -> dict[str, float] | None:
    """Align model ``classes_`` with one row of predict_proba for diagnostics."""
    if not raw_proba_row or not classes_list:
        return None
    if len(classes_list) != len(raw_proba_row):
        return {
            f"index_{i}": float(p) for i, p in enumerate(raw_proba_row)
        }
    out: dict[str, float] = {}
    for c, p in zip(classes_list, raw_proba_row, strict=False):
        out[str(c)] = float(p)
    return out


def _neutral_lean_direction(
    classes_list: list[Any] | None,
    raw_proba_row: list[float] | None,
    *,
    threshold: float,
) -> int | None:
    """
    When predicted class maps to flat, lean long/short from P(+1)-P(-1) if classes are known.

    Returns +1, -1, or None (no lean / insufficient structure).
    """
    if not classes_list or not raw_proba_row or len(classes_list) != len(raw_proba_row):
        return None
    p_long = 0.0
    p_short = 0.0
    for c, p in zip(classes_list, raw_proba_row, strict=False):
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if ci == 1:
            p_long += float(p)
        elif ci == -1:
            p_short += float(p)
    diff = p_long - p_short
    if diff > threshold:
        return 1
    if diff < -threshold:
        return -1
    return None


@dataclass
class Phase21SignalMappingParams:
    """Execution-driven knobs (from :class:`config.experiment_schema.ExecutionConfig`)."""

    signal_min_confidence_for_entry: float | None = None
    """Fallback when direction-specific thresholds are unset."""
    signal_min_confidence_for_entry_long: float | None = None
    """If set, long (``direction == 1``) uses this instead of ``signal_min_confidence_for_entry``."""
    signal_min_confidence_for_entry_short: float | None = None
    """If set, short (``direction == -1``) uses this instead of ``signal_min_confidence_for_entry``."""
    signal_flat_if_max_proba_below: float | None = None
    signal_allow_neutral_class: bool = False
    signal_neutral_lean_threshold: float = 0.0
    signal_class_mapping_override: dict[int, int] | None = None


def _min_confidence_for_direction(direction: int, params: Phase21SignalMappingParams) -> float | None:
    """Resolve entry confidence floor for a directional intent (asymmetric long vs short)."""
    if direction == 1 and params.signal_min_confidence_for_entry_long is not None:
        return float(params.signal_min_confidence_for_entry_long)
    if direction == -1 and params.signal_min_confidence_for_entry_short is not None:
        return float(params.signal_min_confidence_for_entry_short)
    return params.signal_min_confidence_for_entry


def apply_phase21_direction_pipeline(
    *,
    pred: int,
    proba: np.ndarray | None,
    raw_proba_list: list[float] | None,
    classes_list: list[Any] | None,
    research_class_to_direction: dict[int, int] | None,
    params: Phase21SignalMappingParams,
) -> tuple[int, dict[str, Any]]:
    """
    Compute final direction and a diagnostic trace (post-rules applied on top of base mapping).

    Order:
    1. Merge class maps; base direction from class (with optional neutral-class lean).
    2. ``signal_flat_if_max_proba_below`` — max proba below → flat.
    3. Directional confidence gate — non-flat requires max proba ≥ threshold for that side
       (``signal_min_confidence_for_entry_long`` / ``_short`` when set, else
       ``signal_min_confidence_for_entry``).
    """
    trace: dict[str, Any] = {
        "phase21_pipeline_version": 2,
        "confidence_thresholds": {
            "signal_min_confidence_for_entry": params.signal_min_confidence_for_entry,
            "signal_min_confidence_for_entry_long": params.signal_min_confidence_for_entry_long,
            "signal_min_confidence_for_entry_short": params.signal_min_confidence_for_entry_short,
            "signal_flat_if_max_proba_below": params.signal_flat_if_max_proba_below,
        },
        "signal_allow_neutral_class": params.signal_allow_neutral_class,
        "signal_neutral_lean_threshold": params.signal_neutral_lean_threshold,
    }

    effective_map = merge_class_direction_maps(
        research_class_to_direction,
        params.signal_class_mapping_override,
    )
    trace["effective_class_to_direction"] = effective_map

    mapping_kind = classify_direction_mapping(pred, effective_map)
    trace["direction_mapping_rule"] = mapping_kind

    direction = default_direction_from_class(int(pred), effective_map)

    lean_applied = False
    if (
        params.signal_allow_neutral_class
        and direction == 0
        and mapping_kind == "triple_barrier_identity"
        and int(pred) == 0
    ):
        lean = _neutral_lean_direction(
            classes_list,
            raw_proba_list,
            threshold=float(params.signal_neutral_lean_threshold),
        )
        if lean is not None:
            direction = lean
            lean_applied = True
            trace["neutral_class_lean_direction"] = lean

    trace["direction_after_class_mapping"] = int(direction)
    trace["neutral_lean_applied"] = lean_applied

    max_p: float | None = None
    if raw_proba_list:
        max_p = float(max(raw_proba_list))

    if max_p is not None:
        trace["max_class_probability"] = max_p

    flat_floor = params.signal_flat_if_max_proba_below
    if flat_floor is not None and max_p is not None and max_p < float(flat_floor):
        trace["flat_gate_applied"] = "signal_flat_if_max_proba_below"
        trace["direction_before_flat_gate"] = int(direction)
        direction = 0

    min_entry = _min_confidence_for_direction(int(direction), params)
    if min_entry is not None and direction != 0:
        trace["entry_confidence_threshold_applied"] = float(min_entry)
        if max_p is None:
            trace["entry_confidence_gate"] = "skipped_no_proba"
        elif max_p < float(min_entry):
            trace["entry_confidence_gate"] = "blocked_below_signal_min_confidence_for_entry"
            trace["direction_before_entry_gate"] = int(direction)
            direction = 0
        else:
            trace["entry_confidence_gate"] = "passed"

    trace["final_direction"] = int(direction)
    trace["per_class_probability"] = per_class_probabilities(classes_list, raw_proba_list)
    trace["final_direction_reason"] = _final_direction_reason(
        direction=int(direction),
        pred=int(pred),
        mapping_kind=mapping_kind,
        effective_map=effective_map,
        trace=trace,
    )
    trace["neutral_flat_explanation"] = neutral_flat_explanation(
        direction=direction,
        primary_class=pred,
        mapping_kind=mapping_kind,
        class_to_direction=effective_map,
    )
    if direction == 0:
        extra = trace.get("flat_gate_applied") or trace.get("entry_confidence_gate")
        if extra:
            prev = trace["neutral_flat_explanation"] or ""
            trace["neutral_flat_explanation"] = (
                f"{prev} Additional gate: {extra}."
            ).strip()

    return int(direction), trace


def _final_direction_reason(
    *,
    direction: int,
    pred: int,
    mapping_kind: str,
    effective_map: dict[int, int] | None,
    trace: dict[str, Any],
) -> str:
    if trace.get("flat_gate_applied"):
        return "flat_uncertainty_max_proba_below_threshold"
    if trace.get("entry_confidence_gate") == "blocked_below_signal_min_confidence_for_entry":
        return "flat_entry_confidence_below_signal_min_confidence_for_entry"
    if trace.get("neutral_lean_applied"):
        return "long_or_short_from_neutral_class_probability_lean" if direction != 0 else "flat_after_neutral_lean"
    if direction == 1:
        return "long_from_class_mapping"
    if direction == -1:
        return "short_from_class_mapping"
    if mapping_kind == "triple_barrier_identity" and pred == 0:
        return "flat_model_hold_class_zero"
    if mapping_kind == "generic_classifier_posneg" and pred <= 0:
        return "flat_generic_nonpositive_class"
    if mapping_kind == "explicit_class_to_direction" and effective_map is not None:
        return f"flat_explicit_map_class_{pred}_to_zero"
    return "flat_after_mapping"
