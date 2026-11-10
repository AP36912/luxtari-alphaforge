"""
Model configuration for registry-based construction.

Validation splits and backtesting stay out of this module (Phase 6+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """
    Select and parameterize a model by name.

    ``extra_params`` are passed to the underlying estimator / torch module
    (e.g. ``n_estimators``, ``max_depth``, ``hidden_size``).
    """

    name: str
    """Registry key, e.g. ``lightgbm``, ``logistic_regression``, ``lstm``."""

    extra_params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the concrete model constructor."""
