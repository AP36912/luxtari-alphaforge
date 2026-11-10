"""
Experiment tracking and hyperparameter search.

Import concrete entry points to avoid loading the full graph at package import, e.g.:

- ``from tracking.store import ExperimentStore``
- ``from tracking.search import run_parameter_search``
- ``from tracking.record import log_experiment_run``
- ``TrackingConfig`` lives in ``config.experiment_schema`` (used by ``ExperimentConfig``).
"""
