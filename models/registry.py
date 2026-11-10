"""
Model registry: construct and load models from name or :class:`ModelConfig`.
"""

from __future__ import annotations

from pathlib import Path

from models.artifact_trust import (
    get_model_artifact_trust_policy,
    prepare_trusted_model_dir,
)
from models.base import BaseQuantModel
from models.config import ModelConfig
from models.persistence import read_meta

_NAME_ALIASES: dict[str, str] = {
    "rf": "random_forest",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "xgb": "xgboost",
    "log_reg": "logistic_regression",
    "logit": "logistic_regression",
    "cnn1d": "cnn",
}


def _resolve_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return _NAME_ALIASES.get(key, key)


def _registry_map() -> dict[str, type[BaseQuantModel]]:
    """Late-bound to keep optional heavy imports out of import graph until needed."""
    from models.deep_torch import CNN1DClassifierModel, LSTMClassifierModel
    from models.linear import LassoModel, LogisticRegressionModel, RidgeModel
    from models.trees import LightGBMModel, RandomForestModel, XGBoostModel

    return {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "ridge": RidgeModel,
        "lasso": LassoModel,
        "lstm": LSTMClassifierModel,
        "cnn": CNN1DClassifierModel,
    }


def create_model(config: ModelConfig) -> BaseQuantModel:
    """Instantiate a model from :class:`models.config.ModelConfig`."""
    reg = _registry_map()
    name = _resolve_name(config.name)
    if name not in reg:
        keys = ", ".join(sorted(reg))
        raise KeyError(f"Unknown model name {config.name!r} (resolved {name!r}). Known: {keys}")
    return reg[name](**dict(config.extra_params))


def load_model(path: str | Path) -> BaseQuantModel:
    """
    Load any registered model from a directory saved via :meth:`BaseQuantModel.save`.

    Reads ``meta.json`` for ``model_type`` and dispatches to the matching class.
    """
    with prepare_trusted_model_dir(
        Path(path), get_model_artifact_trust_policy()
    ) as artifact:
        snapshot = artifact.snapshot_path
        meta = read_meta(snapshot)
        mtype = meta.get("model_type")
        if not mtype:
            raise ValueError("meta.json missing model_type")
        reg = _registry_map()
        if mtype not in reg:
            raise KeyError(f"Unknown model_type in archive: {mtype!r}")
        return reg[mtype].load(snapshot)


def list_registered_models() -> tuple[str, ...]:
    """Sorted registry keys (after alias resolution targets)."""
    return tuple(sorted(_registry_map().keys()))
