"""
Supervised models: sklearn / boosting / light PyTorch sequences.

Use :func:`models.registry.create_model` with :class:`models.config.ModelConfig`
or :class:`models.training.ModelTrainer` for fit/predict/save/load.

Concrete classes live in ``models.linear``, ``models.trees``, ``models.deep_torch``.
"""

from models.base import BaseModel, BaseQuantModel
from models.config import ModelConfig
from models.registry import create_model, list_registered_models, load_model
from models.training import ModelTrainer

__all__ = [
    "BaseModel",
    "BaseQuantModel",
    "ModelConfig",
    "ModelTrainer",
    "create_model",
    "list_registered_models",
    "load_model",
]
