"""
Position sizing: fixed fraction of equity, volatility targeting, Kelly scaling.

Outputs are **portfolio weights** (dimensionless); multiply by equity and divide
by prices for share counts in execution. Works for any ``n_assets`` (single row).
"""

from __future__ import annotations

import numpy as np


def weights_fixed_fraction(signals: np.ndarray, fraction: float) -> np.ndarray:
    """
    Scale signed signals into weights: ``w_i = clip(s_i, -1, 1) * fraction``.

    ``fraction`` is the per-name cap on absolute weight before global risk clips.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    s = np.asarray(signals, dtype=np.float64).reshape(-1)
    s = np.clip(s, -1.0, 1.0)
    return s * fraction


def portfolio_vol_diagonal(weights: np.ndarray, vol_annual: np.ndarray) -> float:
    """
    Annualized portfolio volatility under **zero correlation** and weights summing arbitrarily.

    ``σ_p = sqrt(sum_i w_i^2 σ_i^2)`` with ``σ_i`` annualized vols of each asset.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    v = np.asarray(vol_annual, dtype=np.float64).reshape(-1)
    if w.shape != v.shape:
        raise ValueError("weights and vol_annual must have same shape")
    return float(np.sqrt(np.sum((w**2) * (v**2))))


def weights_volatility_target(
    signals: np.ndarray,
    vol_annual: np.ndarray,
    target_vol_annual: float,
    *,
    max_gross: float = 1.0,
    min_vol: float = 1e-6,
) -> np.ndarray:
    """
    Inverse-volatility tilt, scaled toward ``target_vol_annual`` (diagonal model).

    Steps:
        1. ``raw_i = clip(s_i) / max(σ_i, min_vol)``
        2. Normalize to gross exposure 1 (if possible).
        3. Scale so diagonal portfolio vol equals ``target_vol`` (if ``σ_p > 0``).
        4. Clip gross to ``max_gross``.
    """
    if target_vol_annual <= 0:
        raise ValueError("target_vol_annual must be positive")
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")

    s = np.clip(np.asarray(signals, dtype=np.float64).reshape(-1), -1.0, 1.0)
    sigma = np.asarray(vol_annual, dtype=np.float64).reshape(-1)
    if s.shape != sigma.shape:
        raise ValueError("signals and vol_annual must match")

    inv = np.where(sigma > min_vol, 1.0 / sigma, 0.0)
    raw = s * inv
    g = np.sum(np.abs(raw))
    if g < 1e-15:
        return np.zeros_like(s)

    w = raw / g
    sigma_p = portfolio_vol_diagonal(w, sigma)
    if sigma_p < min_vol:
        w_scaled = w * max_gross
    else:
        scale = target_vol_annual / sigma_p
        w_scaled = w * scale

    gg = np.sum(np.abs(w_scaled))
    if gg > max_gross > 0:
        w_scaled = w_scaled * (max_gross / gg)
    return w_scaled


def scale_weights_by_kelly(weights: np.ndarray, kelly_scale: float) -> np.ndarray:
    """Multiply all weights by a safe Kelly fraction (typically ``<= 0.25``)."""
    if kelly_scale < 0:
        raise ValueError("kelly_scale must be non-negative")
    return np.asarray(weights, dtype=np.float64) * float(kelly_scale)
