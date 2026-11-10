"""
Scikit-learn linear models: logistic (classification), Ridge/Lasso (regression).

``predict_proba`` is available only for logistic regression (multiclass supported).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from models.artifact_trust import get_model_artifact_trust_policy, trusted_model_dir
from models.base import BaseQuantModel
from models.persistence import model_dir, read_meta, write_meta


def _require_trust(path: str | Path):
    """Yield a verified snapshot for direct or registry-dispatched loads."""
    return trusted_model_dir(path, get_model_artifact_trust_policy())


class _SklearnWrapper(BaseQuantModel):
    """Shared joblib persistence for sklearn estimators."""

    model_type: str = "sklearn_generic"
    _EstimatorClass: type = Ridge

    def __init__(
        self,
        *,
        _fitted_estimator: Any | None = None,
        _stored_init_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if _fitted_estimator is not None:
            self._estimator = _fitted_estimator
            self._init_kwargs = dict(_stored_init_kwargs or {})
        else:
            self._init_kwargs = self._merge_init_kwargs(kwargs)
            self._estimator = self._EstimatorClass(**self._init_kwargs)

    def _merge_init_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return dict(kwargs)

    def save(self, path: str | Path) -> None:
        d = model_dir(path)
        joblib.dump(self._estimator, d / "estimator.joblib")
        write_meta(
            d,
            {
                "model_type": self.model_type,
                "init_kwargs": self._init_kwargs,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> BaseQuantModel:
        with _require_trust(path) as d:
            meta = read_meta(d)
            if meta.get("model_type") != cls.model_type:
                raise ValueError(
                    f"Expected model_type {cls.model_type!r}, got {meta.get('model_type')!r}"
                )
            est = joblib.load(d / "estimator.joblib")
            return cls(
                _fitted_estimator=est,
                _stored_init_kwargs=meta.get("init_kwargs", {}),
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        self._estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X))

    @property
    def classes_(self) -> np.ndarray | None:
        if hasattr(self._estimator, "classes_"):
            return np.asarray(self._estimator.classes_)
        return None


class LogisticRegressionModel(_SklearnWrapper):
    supports_proba = True
    model_type = "logistic_regression"
    _EstimatorClass = LogisticRegression  # type: ignore[assignment]

    def _merge_init_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        defaults = {"max_iter": 1000, "solver": "lbfgs"}
        return {**defaults, **kwargs}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict_proba(X))


class RidgeModel(_SklearnWrapper):
    supports_proba = False
    model_type = "ridge"
    _EstimatorClass = Ridge  # type: ignore[assignment]


class LassoModel(_SklearnWrapper):
    supports_proba = False
    model_type = "lasso"
    _EstimatorClass = Lasso  # type: ignore[assignment]
