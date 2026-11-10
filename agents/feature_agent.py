"""
Feature agent: build features via ``features.pipeline`` and summarize selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from agents.types import FeatureAgentOutput, FeatureAgentStepResult
from config.experiment_schema import feature_config_from_overrides
from features.pipeline import build_features, feature_column_names


@dataclass
class FeatureAgentInput:
    ohlcv: pd.DataFrame
    feature_overrides: dict[str, Any]
    dropna: bool = True
    include_ohlcv: bool = True


def run_feature_agent(inp: FeatureAgentInput) -> FeatureAgentStepResult:
    """
    Apply :func:`build_features` with resolved :class:`FeatureConfig` and return summaries.

    ``feature_config_effective`` is a JSON-friendly view of overridden fields only
    (defaults are implicit in :class:`FeatureConfig`).
    """
    cfg = feature_config_from_overrides(inp.feature_overrides or None)
    n_before = len(inp.ohlcv)
    featured = build_features(
        inp.ohlcv,
        cfg,
        dropna=inp.dropna,
        include_ohlcv=inp.include_ohlcv,
    )
    names = tuple(feature_column_names(featured))
    eff = asdict(cfg)
    out = FeatureAgentOutput(
        feature_names=names,
        n_rows=len(featured),
        n_dropped_na=max(0, n_before - len(featured)) if inp.dropna else 0,
        feature_config_effective=eff,
    )
    return FeatureAgentStepResult(output=out, featured=featured)
