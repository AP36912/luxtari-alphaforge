"""
Abstract model interface for supervised quant models.

Callers supply PIT-safe feature matrices; this layer does not read OHLCV directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseQuantModel(ABC):
    """
    Minimal contract: fit, predict, optional ``predict_proba``, persistence.

    Subclasses set ``supports_proba`` to True when class probabilities exist.
    """

    supports_proba: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        """Train on 2D ``X`` (n_samples, n_features) and 1D ``y``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions; same number of rows as ``X`` unless documented otherwise."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities, shape ``(n_samples, n_classes)``, sorted by ``classes_``.

        Raises:
            NotImplementedError: if ``supports_proba`` is False.
        """
        if not self.supports_proba:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement predict_proba"
            )
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model to a path (file or directory, implementation-defined)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseQuantModel:
        """Load a model previously saved with :meth:`save`."""
        raise NotImplementedError

    @property
    def classes_(self) -> np.ndarray | None:
        """Unique training labels for classifiers; ``None`` for regressors."""
        return None


# Backward-compatible alias
BaseModel = BaseQuantModel
