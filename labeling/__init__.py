"""Labeling package — triple-barrier / meta-label only."""

from __future__ import annotations

from labeling.config import MetaLabelConfig, TripleBarrierConfig
from labeling.engine import LabelingEngine
from labeling.meta_label import apply_meta_label
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels

__all__ = [
    "MetaLabelConfig",
    "TripleBarrierConfig",
    "LabelingEngine",
    "apply_meta_label",
    "compute_triple_barrier_labels",
    "merge_labels",
]
