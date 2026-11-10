"""Aggregate statistics on classifier predicted probabilities (test segments)."""

from __future__ import annotations

from typing import Any

import numpy as np


def prediction_probability_stats(proba: np.ndarray) -> dict[str, Any]:
    """
    ``proba`` shape ``(n_samples, n_classes)``, rows sum ~1.

    ``fraction_near_uniform`` uses max proba in [0.42, 0.58] as a 50/50 band proxy.
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return {
            "n_samples": int(p.shape[0]) if p.ndim == 2 else 0,
            "mean_max_class_probability": None,
            "std_max_class_probability": None,
            "mean_top2_margin": None,
            "std_top2_margin": None,
            "fraction_max_proba_0.42_to_0.58": None,
            "mean_entropy": None,
        }
    max_p = p.max(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    ent = -np.sum(p * np.log(np.clip(p, 1e-15, 1.0)), axis=1)
    near_half = np.mean((max_p >= 0.42) & (max_p <= 0.58))
    return {
        "n_samples": int(p.shape[0]),
        "mean_max_class_probability": float(np.mean(max_p)),
        "std_max_class_probability": float(np.std(max_p)),
        "mean_top2_margin": float(np.mean(margin)),
        "std_top2_margin": float(np.std(margin)),
        "fraction_max_proba_0.42_to_0.58": float(near_half),
        "mean_entropy": float(np.mean(ent)),
    }


def merge_weighted_signal_stats(
    acc: dict[str, float],
    stats: dict[str, Any],
    weight: int,
) -> None:
    """Accumulate weighted means for streaming aggregate (ignores None)."""
    n = stats.get("n_samples") or 0
    if not n:
        return
    w = float(weight)
    for key in (
        "mean_max_class_probability",
        "mean_top2_margin",
        "fraction_max_proba_0.42_to_0.58",
        "mean_entropy",
    ):
        v = stats.get(key)
        if v is None:
            continue
        acc["_w_" + key] = acc.get("_w_" + key, 0.0) + w
        acc[key] = acc.get(key, 0.0) + float(v) * w


def finalize_weighted_signal_stats(acc: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "mean_max_class_probability",
        "mean_top2_margin",
        "fraction_max_proba_0.42_to_0.58",
        "mean_entropy",
    ):
        wk = "_w_" + key
        if wk in acc and acc[wk] > 0:
            out[key] = round(acc[key] / acc[wk], 8)
    return out


__all__ = [
    "prediction_probability_stats",
    "merge_weighted_signal_stats",
    "finalize_weighted_signal_stats",
]
