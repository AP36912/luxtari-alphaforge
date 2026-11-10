"""
Labeling engine: triple barriers plus optional meta-label layer.

Use :meth:`LabelingEngine.generate` to produce label columns aligned row-wise
with the input frame (same length, same index order after ``reset_index``).
"""

from __future__ import annotations

import pandas as pd

from labeling.config import MetaLabelConfig, TripleBarrierConfig
from labeling.meta_label import apply_meta_label
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels


class LabelingEngine:
    """Coordinates triple-barrier labeling and optional meta-labels."""

    def __init__(
        self,
        triple_barrier_config: TripleBarrierConfig,
        meta_config: MetaLabelConfig | None = None,
    ) -> None:
        self.triple_barrier_config = triple_barrier_config
        self.meta_config = meta_config

    def generate(
        self,
        frame: pd.DataFrame,
        *,
        attach: bool = False,
    ) -> pd.DataFrame:
        """
        Produce label columns for each row of ``frame``.

        Args:
            frame: OHLCV (+ optional features) DataFrame.
            attach: If True, return ``frame`` concatenated with labels; else labels only.
        """
        labels = compute_triple_barrier_labels(frame, self.triple_barrier_config)
        if self.meta_config is not None:
            labels = labels.copy()
            labels["meta_label"] = apply_meta_label(
                labels["triple_barrier_label"], self.meta_config
            )
        if attach:
            return merge_labels(frame, labels)
        return labels
