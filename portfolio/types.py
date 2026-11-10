"""
Multi-asset extension hooks: ordered universe, weight vectors, batched panels.

All portfolio math uses **numpy**; ``AssetUniverse`` maps names to indices for future
DataFrame joins and multi-asset backtests without changing function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssetUniverse:
    """Ordered instrument identifiers (``weights[i]`` aligns with ``ids[i]``)."""

    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Asset ids must be unique")
        if len(self.ids) == 0:
            raise ValueError("Universe must be non-empty")

    def __len__(self) -> int:
        return len(self.ids)

    def index(self, asset_id: str) -> int:
        try:
            return self.ids.index(asset_id)
        except ValueError as e:
            raise KeyError(asset_id) from e


@dataclass(frozen=True)
class PortfolioWeights:
    """Container for parallel ``ids`` and ``values`` (gross exposure = sum abs(values))."""

    universe: AssetUniverse
    values: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if v.shape[0] != len(self.universe):
            raise ValueError("values length must match universe size")
        object.__setattr__(self, "values", v)

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.values)))

    @property
    def net_exposure(self) -> float:
        return float(np.sum(self.values))

    def as_dict(self) -> dict[str, float]:
        return {a: float(w) for a, w in zip(self.universe.ids, self.values)}


def zeros_like_universe(universe: AssetUniverse) -> np.ndarray:
    return np.zeros(len(universe), dtype=np.float64)


def validate_square_cov(cov: np.ndarray, n: int) -> np.ndarray:
    """Return float64 (n,n) covariance or raise."""
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (n, n):
        raise ValueError(f"covariance must be ({n},{n}), got {c.shape}")
    return c
