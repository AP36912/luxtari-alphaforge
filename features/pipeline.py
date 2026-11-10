"""
Feature pipeline: compose causal blocks, align timestamps, drop NaNs.

Public candidate excludes proprietary alpha packs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS
from features.config import FeatureConfig
from features.discovery import add_discovery_features, discovery_feature_names
from features.microstructure import add_microstructure_features
from features.price import add_price_features
from features.statistical import add_statistical_features
from features.technical import add_technical_features

logger = logging.getLogger(__name__)

_CORE_BUILDERS: tuple[tuple[str, Callable[[pd.DataFrame, FeatureConfig], pd.DataFrame]], ...] = (
    ("price_enabled", add_price_features),
    ("technical_enabled", add_technical_features),
    ("statistical_enabled", add_statistical_features),
    ("microstructure_enabled", add_microstructure_features),
)


def feature_column_names(df: pd.DataFrame) -> list[str]:
    """Return columns not in the canonical OHLCV schema."""
    return [c for c in df.columns if c not in SCHEMA_COLUMNS]


def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    dropna: bool = True,
    include_ohlcv: bool = True,
) -> pd.DataFrame:
    """Build the public feature matrix (core blocks + optional discovery)."""
    out = df.copy()
    if "timestamp" not in out.columns:
        raise KeyError("Expected column 'timestamp'")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    for flag, builder in _CORE_BUILDERS:
        if bool(getattr(cfg, flag, True)):
            out = builder(out, cfg)
    if cfg.discovery_enabled:
        out = add_discovery_features(out, cfg)

    feats = feature_column_names(out)
    disc = discovery_feature_names(feats)
    if cfg.discovery_enabled and disc:
        logger.info("Discovery features (%d): %s", len(disc), ", ".join(disc))

    if cfg.drop_feature_patterns:
        drop = [c for c in feats if any(pat in c for pat in cfg.drop_feature_patterns)]
        if drop:
            out = out.drop(columns=drop)
            feats = feature_column_names(out)
            logger.info(
                "Pruned %d features matching %s → %d remaining",
                len(drop),
                cfg.drop_feature_patterns,
                len(feats),
            )

    if dropna and feats:
        out = out.dropna(subset=feats, how="any")

    ohlcv_present = [c for c in SCHEMA_COLUMNS if c in out.columns]
    feat_sorted = sorted(c for c in out.columns if c not in SCHEMA_COLUMNS)
    if include_ohlcv:
        out = out[ohlcv_present + feat_sorted]
    else:
        out = out[["timestamp"] + feat_sorted]
    return out.reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Alias for :func:`build_features` with default config."""
    return build_features(df, cfg or FeatureConfig())


class FeatureBlockStep:
    def __init__(
        self,
        fn: Callable[[pd.DataFrame, FeatureConfig], pd.DataFrame],
        cfg: FeatureConfig,
    ) -> None:
        self._fn = fn
        self._cfg = cfg

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._fn(frame, self._cfg)


class FeaturePipeline:
    def __init__(self, steps: list[Any] | None = None) -> None:
        self._steps: list[Any] = list(steps or [])

    def add(self, step: Any) -> None:
        self._steps.append(step)

    def transform(self, frame: Any) -> Any:
        out = frame
        for step in self._steps:
            out = step.transform(out)
        return out


def pipeline_from_config(cfg: FeatureConfig) -> FeaturePipeline:
    pipe = FeaturePipeline()
    for flag, builder in _CORE_BUILDERS:
        if bool(getattr(cfg, flag, True)):
            pipe.add(FeatureBlockStep(builder, cfg))
    if cfg.discovery_enabled:
        pipe.add(FeatureBlockStep(add_discovery_features, cfg))
    return pipe


def finalize_feature_frame(
    df: pd.DataFrame,
    *,
    dropna: bool = True,
    include_ohlcv: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    feats = feature_column_names(out)
    if dropna and feats:
        out = out.dropna(subset=feats, how="any")
    ohlcv_present = [c for c in SCHEMA_COLUMNS if c in out.columns]
    feat_sorted = sorted(c for c in out.columns if c not in SCHEMA_COLUMNS)
    if include_ohlcv:
        out = out[ohlcv_present + feat_sorted]
    else:
        out = out[["timestamp"] + feat_sorted]
    return out.reset_index(drop=True)
