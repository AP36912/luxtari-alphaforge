"""
Market regime and session context (XAUUSD-oriented defaults, configurable).

Use :func:`regime.integration.attach_regime_columns` to add tags/features to a DataFrame;
use :func:`regime.masks.build_holdout_segment_masks` with :mod:`backtest.segmentation` for reports.
"""

from regime.config import (
    RegimeConfig,
    RegimeDetectionParams,
    RegimeFilterConfig,
    SessionCalendarConfig,
    SessionWindow,
    regime_config_from_dict,
)
from regime.integration import attach_regime_columns, regime_feature_column_names
from regime.masks import build_holdout_segment_masks

__all__ = [
    "RegimeConfig",
    "RegimeDetectionParams",
    "RegimeFilterConfig",
    "SessionCalendarConfig",
    "SessionWindow",
    "attach_regime_columns",
    "build_holdout_segment_masks",
    "regime_config_from_dict",
    "regime_feature_column_names",
]
