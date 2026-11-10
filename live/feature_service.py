"""
Live feature computation using the research :func:`features.pipeline.build_features`.

Maintains a rolling OHLCV history and returns the **latest** feature row aligned
to the trained model's column order.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from config.experiment_schema import feature_config_from_overrides
from features.config import FeatureConfig
from features.pipeline import build_features
from regime.config import RegimeConfig, regime_config_from_dict
from regime.integration import attach_regime_columns, regime_feature_column_names


def _max_window_from_feature_config(cfg: FeatureConfig) -> int:
    """Conservative minimum history length (bars) before feature rows are non-NaN."""
    candidates: list[int] = []
    for f in fields(FeatureConfig):
        val = getattr(cfg, f.name)
        if isinstance(val, tuple) and val and isinstance(val[0], int):
            candidates.append(max(val))
        elif isinstance(val, int) and f.name != "volatility_annualization_factor":
            if "factor" in f.name:
                continue
            candidates.append(int(val))
    return max(candidates) + 5 if candidates else 60


@dataclass
class LiveFeatureServiceConfig:
    """Wiring for causal live features + regime columns matching training."""

    feature_overrides: dict[str, Any]
    regime: RegimeConfig
    feature_column_order: list[str]
    """Exact names passed to ``model.predict`` (base features + regime ids if used)."""
    min_history_bars: int | None = None
    """Override warmup if you trained with a custom pipeline."""


class LiveFeatureService:
    """
    Computes features on a rolling buffer; last row must be fully warm (no NaNs
    in modeling columns after pipeline ``dropna`` logic).
    """

    def __init__(self, cfg: LiveFeatureServiceConfig) -> None:
        self._fcfg = feature_config_from_overrides(cfg.feature_overrides)
        self._rcfg = cfg.regime
        self._columns = list(cfg.feature_column_order)
        self._min_bars = cfg.min_history_bars or _max_window_from_feature_config(self._fcfg)
        self._buffer: pd.DataFrame = pd.DataFrame()

    @property
    def min_history_bars(self) -> int:
        return int(self._min_bars)

    @property
    def buffer(self) -> pd.DataFrame:
        return self._buffer.copy()

    def set_buffer(self, ohlcv: pd.DataFrame) -> None:
        """Replace rolling history (caller merges feeds)."""
        self._buffer = ohlcv.reset_index(drop=True).copy()

    def compute_feature_matrix(self) -> pd.DataFrame:
        """
        Run research pipeline on the full buffer, attach regime columns, drop NaNs.

        Returns rows usable for training-parity inference (typically the tail).
        """
        if len(self._buffer) == 0:
            raise ValueError("empty buffer")
        if len(self._buffer) < self._min_bars:
            raise ValueError(
                f"need at least {self._min_bars} bars for feature warmup; got {len(self._buffer)}"
            )
        feats = build_features(self._buffer, self._fcfg, dropna=True, include_ohlcv=True)
        if self._rcfg.enabled:
            feats = attach_regime_columns(feats, self._rcfg, fit_end_index=len(feats))
        return feats.reset_index(drop=True)

    def latest_model_vector(self) -> tuple[np.ndarray, pd.Series]:
        """
        Return ``(X_row, metadata_row)`` for the most recent timestamp.

        Raises if any expected model column is missing.
        """
        mat = self.compute_feature_matrix()
        if mat.empty:
            raise ValueError("feature matrix empty after pipeline (insufficient warmup)")
        row = mat.iloc[-1]
        missing = [c for c in self._columns if c not in mat.columns]
        if missing:
            raise KeyError(f"feature_column_order missing columns in frame: {missing[:10]}…")
        x = np.asarray(row[self._columns].tolist(), dtype=np.float64).reshape(1, -1)
        return x, row

    @staticmethod
    def expected_regime_feature_names(regime_cfg: RegimeConfig) -> list[str]:
        return regime_feature_column_names(regime_cfg)

    @staticmethod
    def infer_feature_column_order_from_training(
        base_feature_names: list[str],
        regime_cfg: RegimeConfig,
    ) -> list[str]:
        """Match experiment pipeline: base features then regime id columns."""
        return list(base_feature_names) + regime_feature_column_names(regime_cfg)
