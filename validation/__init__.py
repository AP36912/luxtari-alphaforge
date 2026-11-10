"""Cross-validation, purged splits, embargo gaps, and classification metrics."""

from validation.event_bounds import event_bounds_from_horizon, intervals_overlap
from validation.metrics import accuracy, precision_recall, roc_auc
from validation.purge import apply_embargo, purge_overlapping_train
from validation.purged_kfold import PurgedKFold
from validation.splits import purged_kfold_indices
from validation.time_series_split import PurgedTimeSeriesSplit
from validation.walk_forward import WalkForwardSplitter

__all__ = [
    "PurgedKFold",
    "PurgedTimeSeriesSplit",
    "WalkForwardSplitter",
    "accuracy",
    "apply_embargo",
    "event_bounds_from_horizon",
    "intervals_overlap",
    "precision_recall",
    "purge_overlapping_train",
    "purged_kfold_indices",
    "roc_auc",
]
