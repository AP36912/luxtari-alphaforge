"""
Central hyperparameters for feature windows.

All windows are backward-looking only. Proprietary alpha packs are not part of
the public candidate; discovery defaults to off.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Config-driven rolling spans and indicator settings."""

    # Price
    return_periods: tuple[int, ...] = (1, 3, 5, 10, 20)
    log_return_periods: tuple[int, ...] = (1, 5)
    volatility_windows: tuple[int, ...] = (10, 20)
    volatility_annualization_factor: float = 252.0
    atr_period: int = 14
    momentum_periods: tuple[int, ...] = (10, 20)
    trend_slope_windows: tuple[int, ...] = (20, 50)

    # Technical
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    ma_windows: tuple[int, ...] = (10, 50)
    ma_types: tuple[str, ...] = ("sma", "ema")
    ma_slope_windows: tuple[int, ...] = (20,)

    # Statistical
    zscore_windows: tuple[int, ...] = (20,)
    zscore_columns: tuple[str, ...] = ("close",)
    rolling_variance_windows: tuple[int, ...] = (20,)
    skew_windows: tuple[int, ...] = (30,)
    kurtosis_windows: tuple[int, ...] = (30,)

    # Microstructure proxies (bar-level)
    tick_direction_window: int = 5
    volume_imbalance_window: int = 10
    volatility_cluster_short: int = 5
    volatility_cluster_long: int = 20

    # Pipeline
    feature_prefix: str = ""

    # Optional discovery block (generic; off by default in public edition)
    discovery_enabled: bool = False
    discovery_ret_stats_window: int = 20
    discovery_atr_pct_window: int = 100
    discovery_adx_period: int = 14
    discovery_dir_consistency_window: int = 10
    discovery_position_ema_span: int = 50
    discovery_session_timezone: str = "UTC"
    discovery_session_assume_utc_naive: bool = True
    discovery_trend_slope_window: int = 0

    # Feature pruning
    drop_feature_patterns: tuple[str, ...] = ()

    # Block enable flags
    price_enabled: bool = True
    technical_enabled: bool = True
    statistical_enabled: bool = True
    microstructure_enabled: bool = True

    def prefixed(self, name: str) -> str:
        p = self.feature_prefix.strip()
        return f"{p}_{name}" if p else name
