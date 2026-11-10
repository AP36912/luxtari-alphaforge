"""
Thin training facade: fit / predict / predict_proba / save / load.

Delegates to a :class:`models.base.BaseQuantModel` instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseQuantModel
from models.registry import load_model


class ModelTrainer:
    """Wraps a :class:`BaseQuantModel` for a uniform training/inference API."""

    def __init__(self, model: BaseQuantModel) -> None:
        self._model = model

    @property
    def model(self) -> BaseQuantModel:
        return self._model

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> ModelTrainer:
        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        self._model.save(path)

    @classmethod
    def load(cls, path: str | Path) -> ModelTrainer:
        return cls(load_model(path))
