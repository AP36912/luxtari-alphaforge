"""
Tree ensembles: LightGBM, XGBoost, scikit-learn Random Forest.

All expose class probabilities via ``predict_proba`` for classification labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from models.artifact_trust import get_model_artifact_trust_policy, trusted_model_dir
from models.base import BaseQuantModel
from models.persistence import model_dir, read_meta, write_meta


def _require_trust(path: str | Path):
    """Yield a verified snapshot for direct or registry-dispatched loads."""
    return trusted_model_dir(path, get_model_artifact_trust_policy())


class _SklearnForestLike(BaseQuantModel):
    supports_proba = True
    model_type = "random_forest"

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
            self._init_kwargs = dict(kwargs)
            self._estimator = RandomForestClassifier(**self._init_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        self._estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict_proba(X))

    @property
    def classes_(self) -> np.ndarray | None:
        return np.asarray(self._estimator.classes_)

    def save(self, path: str | Path) -> None:
        d = model_dir(path)
        joblib.dump(self._estimator, d / "estimator.joblib")
        write_meta(
            d,
            {"model_type": self.model_type, "init_kwargs": self._init_kwargs},
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


class RandomForestModel(_SklearnForestLike):
    model_type = "random_forest"


class LightGBMModel(BaseQuantModel):
    supports_proba = True
    model_type = "lightgbm"

    def __init__(
        self,
        *,
        _fitted_estimator: Any | None = None,
        _stored_init_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from lightgbm import LGBMClassifier

        if _fitted_estimator is not None:
            self._estimator = _fitted_estimator
            self._init_kwargs = dict(_stored_init_kwargs or {})
        else:
            defaults = {
                "n_estimators": 50,
                "max_depth": -1,
                "learning_rate": 0.05,
                "verbosity": -1,
                "force_col_wise": True,
            }
            self._init_kwargs = {**defaults, **kwargs}
            self._estimator = LGBMClassifier(**self._init_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        self._estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict_proba(X))

    @property
    def classes_(self) -> np.ndarray | None:
        return np.asarray(self._estimator.classes_)

    def save(self, path: str | Path) -> None:
        d = model_dir(path)
        joblib.dump(self._estimator, d / "estimator.joblib")
        write_meta(
            d,
            {"model_type": self.model_type, "init_kwargs": self._init_kwargs},
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


class XGBoostModel(BaseQuantModel):
    supports_proba = True
    model_type = "xgboost"

    def __init__(
        self,
        *,
        _fitted_estimator: Any | None = None,
        _stored_init_kwargs: dict[str, Any] | None = None,
        _label_encoder: LabelEncoder | None = None,
        **kwargs: Any,
    ) -> None:
        from xgboost import XGBClassifier

        self._le = _label_encoder
        if _fitted_estimator is not None:
            self._estimator = _fitted_estimator
            self._init_kwargs = dict(_stored_init_kwargs or {})
        else:
            defaults = {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "verbosity": 0,
            }
            self._init_kwargs = {**defaults, **kwargs}
            self._estimator = XGBClassifier(**self._init_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(np.asarray(y).reshape(-1))
        self._estimator.fit(X, y_enc, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        enc = np.asarray(self._estimator.predict(X), dtype=int)
        assert self._le is not None
        return np.asarray(self._le.inverse_transform(enc))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Columns follow ``self.classes_`` (``LabelEncoder`` / XGBoost 0..K-1 order)."""
        return np.asarray(self._estimator.predict_proba(X))

    @property
    def classes_(self) -> np.ndarray | None:
        if self._le is not None:
            return np.asarray(self._le.classes_)
        if hasattr(self._estimator, "classes_"):
            return np.asarray(self._estimator.classes_)
        return None

    def save(self, path: str | Path) -> None:
        if self._le is None:
            raise RuntimeError("Cannot save unfitted XGBoostModel")
        d = model_dir(path)
        joblib.dump(self._estimator, d / "estimator.joblib")
        joblib.dump(self._le, d / "label_encoder.joblib")
        write_meta(
            d,
            {"model_type": self.model_type, "init_kwargs": self._init_kwargs},
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
            le = joblib.load(d / "label_encoder.joblib")
            return cls(
                _fitted_estimator=est,
                _stored_init_kwargs=meta.get("init_kwargs", {}),
                _label_encoder=le,
            )
