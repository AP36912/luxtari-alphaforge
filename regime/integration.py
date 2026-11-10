"""
Attach regime / session columns to a labeled feature frame and list ML feature names.

Designed for use after triple-barrier merge (or any OHLCV+timestamp frame).
"""

from __future__ import annotations

import pandas as pd

from regime.clustering import fit_predict_regime_clusters
from regime.config import RegimeConfig
from regime.detection import build_rule_based_regime_frame, compute_regime_feature_matrix, structure_to_id
from regime.session_context import compute_session_tags, session_tag_to_id
from regime.volatility_regime import vol_state_to_id


def regime_feature_column_names(cfg: RegimeConfig) -> list[str]:
    """Columns to append to the experiment feature matrix when ``add_feature_columns``."""
    if not cfg.enabled or not cfg.add_feature_columns:
        return []
    cols = ["regime_session_id", "regime_vol_state_id", "regime_structure_id"]
    if cfg.detection_method == "cluster_offline":
        cols.append("regime_cluster_id")
    return cols


def attach_regime_columns(
    df: pd.DataFrame,
    cfg: RegimeConfig,
    *,
    fit_end_index: int | None = None,
) -> pd.DataFrame:
    """
    Add string tags and optional integer features for sessions, vol/structure regimes, clustering.

    Args:
        df: Must include ``timestamp``, ``open``, ``high``, ``low``, ``close``.
        fit_end_index: Training split end for ``cluster_offline`` K-means fit (exclusive).
    """
    if not cfg.enabled:
        return df
    required = ("timestamp", "open", "high", "low", "close")
    for c in required:
        if c not in df.columns:
            raise KeyError(f"attach_regime_columns requires {c!r}")

    out = df.copy()
    out["regime_session"] = compute_session_tags(out, cfg.sessions).values
    rb = build_rule_based_regime_frame(out, cfg.detection)
    out["regime_vol_state"] = rb["regime_vol_state"].values
    out["regime_structure"] = rb["regime_structure"].values

    n = len(out)
    if cfg.detection_method == "cluster_offline":
        F = compute_regime_feature_matrix(out, cfg.detection)
        end = int(fit_end_index if fit_end_index is not None else n)
        out["regime_cluster_id"] = fit_predict_regime_clusters(
            F,
            fit_end_index=end,
            n_clusters=cfg.detection.cluster_n_clusters,
            random_state=cfg.detection.cluster_random_state,
            min_train_rows=cfg.detection.cluster_min_train_rows,
        )
    else:
        out["regime_cluster_id"] = -1

    if cfg.add_feature_columns:
        out["regime_session_id"] = session_tag_to_id(out["regime_session"])
        out["regime_vol_state_id"] = vol_state_to_id(pd.Series(out["regime_vol_state"]))
        out["regime_structure_id"] = structure_to_id(pd.Series(out["regime_structure"]))
    return out


def session_mask_equals(session_series: pd.Series, tag: str) -> pd.Series:
    return session_series.astype(str).eq(tag)
