"""Public candidate config package."""
"""
Configuration package: paths, experiment schema, YAML loading, and full pipeline runner.
"""

from config.experiment_schema import (
    AuditConfig,
    DataConfig,
    ExperimentConfig,
    OutputConfig,
    TrackingConfig,
    ValidationConfig,
    WalkForwardCalibrationConfig,
    experiment_config_from_dict,
    tracking_config_from_dict,
)
from config.pipeline import run_experiment, run_experiment_from_yaml
from config.settings import ProjectPaths, get_project_paths
from config.yaml_loader import load_experiment_yaml

__all__ = [
    "AuditConfig",
    "DataConfig",
    "ExperimentConfig",
    "OutputConfig",
    "ProjectPaths",
    "TrackingConfig",
    "ValidationConfig",
    "WalkForwardCalibrationConfig",
    "experiment_config_from_dict",
    "tracking_config_from_dict",
    "get_project_paths",
    "load_experiment_yaml",
    "run_experiment",
    "run_experiment_from_yaml",
]
