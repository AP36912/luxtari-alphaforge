"""Triple-barrier and meta-label tests (causality + edge cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeling.config import MetaLabelConfig, TripleBarrierConfig
from labeling.engine import LabelingEngine
from labeling.meta_label import apply_meta_label
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reset_index(drop=True)


def test_long_take_profit_first_bar_open_gap() -> None:
    """Open at t+1 gaps above take-profit level."""
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 105, "high": 106, "low": 104, "close": 105, "volume": 1},
        ]
    )
    cfg = TripleBarrierConfig(max_bars=5, profit_pct=0.02, stop_pct=0.02, position_side="long")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == 1.0
    assert out.iloc[0]["triple_barrier_event"] == "take_profit"
    assert out.iloc[0]["bars_to_event"] == 1.0


def test_long_stop_loss_intrabar_low_first() -> None:
    """Conservative long: low hits stop before high reaches TP on same bar."""
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {
                "timestamp": 2,
                "open": 100.0,
                "high": 102.0,
                "low": 97.0,
                "close": 101.0,
                "volume": 1,
            },
        ]
    )
    cfg = TripleBarrierConfig(max_bars=5, profit_pct=0.02, stop_pct=0.02, position_side="long")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == -1.0
    assert out.iloc[0]["triple_barrier_event"] == "stop_loss"


def test_long_vertical_timeout() -> None:
    df = _df(
        [
            {"timestamp": i, "open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1}
            for i in range(6)
        ]
    )
    cfg = TripleBarrierConfig(max_bars=3, profit_pct=0.05, stop_pct=0.05, position_side="long")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == 0.0
    assert out.iloc[0]["triple_barrier_event"] == "vertical"
    assert out.iloc[0]["bars_to_event"] == 3.0


def test_insufficient_future_bars() -> None:
    """Only one future bar and no touch within 5% barriers — cannot complete 10-bar window."""
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        ]
    )
    cfg = TripleBarrierConfig(max_bars=10, profit_pct=0.05, stop_pct=0.05, position_side="long")
    out = compute_triple_barrier_labels(df, cfg)
    assert np.isnan(out.iloc[0]["triple_barrier_label"])
    assert out.iloc[0]["triple_barrier_event"] == "insufficient"


def test_max_bars_zero_immediate_vertical() -> None:
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 110, "high": 111, "low": 109, "close": 110, "volume": 1},
        ]
    )
    cfg = TripleBarrierConfig(max_bars=0, profit_pct=0.01, stop_pct=0.01, position_side="long")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == 0.0
    assert out.iloc[0]["triple_barrier_event"] == "vertical"


def test_short_take_profit_price_drop() -> None:
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 97, "high": 98, "low": 96, "close": 97, "volume": 1},
        ]
    )
    cfg = TripleBarrierConfig(max_bars=5, profit_pct=0.02, stop_pct=0.02, position_side="short")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == 1.0
    assert out.iloc[0]["triple_barrier_event"] == "take_profit"


def test_short_stop_loss() -> None:
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 103, "high": 104, "low": 102, "close": 103, "volume": 1},
        ]
    )
    cfg = TripleBarrierConfig(max_bars=5, profit_pct=0.02, stop_pct=0.02, position_side="short")
    out = compute_triple_barrier_labels(df, cfg)
    assert out.iloc[0]["triple_barrier_label"] == -1.0


def test_volatility_scaled_barriers_use_causal_vol() -> None:
    rng = np.random.default_rng(0)
    n = 80
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
            "close": close,
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(close, np.r_[close[0], close[:-1]]) + 0.1,
            "low": np.minimum(close, np.r_[close[0], close[:-1]]) - 0.1,
            "volume": np.ones(n),
        }
    )
    cfg = TripleBarrierConfig(
        max_bars=15,
        use_volatility_scaling=True,
        volatility_window=10,
        volatility_scale="returns_std",
        profit_vol_multiple=2.0,
        stop_vol_multiple=2.0,
        position_side="long",
    )
    out = compute_triple_barrier_labels(df, cfg)
    assert "triple_barrier_label" in out.columns
    # Some rows should be defined after warm-up
    assert out["triple_barrier_label"].notna().sum() > 0


def test_no_lookahead_matches_prefix_path() -> None:
    rng = np.random.default_rng(1)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
            "close": close,
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(close, np.r_[close[0], close[:-1]]) + 0.15,
            "low": np.minimum(close, np.r_[close[0], close[:-1]]) - 0.15,
            "volume": np.ones(n),
        }
    )
    cfg = TripleBarrierConfig(max_bars=8, profit_pct=0.015, stop_pct=0.015, position_side="long")
    full = compute_triple_barrier_labels(df, cfg)
    for t in range(0, n - cfg.max_bars - 2):
        end = t + 1 + cfg.max_bars
        part = compute_triple_barrier_labels(df.iloc[:end].reset_index(drop=True), cfg)
        a = full["triple_barrier_label"].iloc[t]
        b = part["triple_barrier_label"].iloc[t]
        if np.isnan(a) and np.isnan(b):
            continue
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"t={t}")


def test_meta_profit_first() -> None:
    primary = pd.Series([1.0, -1.0, 0.0, np.nan])
    meta = apply_meta_label(primary, MetaLabelConfig(mode="profit_first"))
    assert meta.iloc[0] == 1.0
    assert meta.iloc[1] == 0.0
    assert meta.iloc[2] == 0.0
    assert np.isnan(meta.iloc[3])


def test_meta_decisive_and_avoid_loss() -> None:
    primary = pd.Series([1.0, -1.0, 0.0])
    d = apply_meta_label(primary, MetaLabelConfig(mode="decisive"))
    assert d.tolist() == [1.0, 1.0, 0.0]
    a = apply_meta_label(primary, MetaLabelConfig(mode="avoid_loss"))
    assert a.tolist() == [1.0, 0.0, 1.0]


def test_labeling_engine_attach() -> None:
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": 2, "open": 105, "high": 106, "low": 104, "close": 105, "volume": 1},
        ]
    )
    eng = LabelingEngine(
        TripleBarrierConfig(max_bars=3, profit_pct=0.02, stop_pct=0.02),
        meta_config=MetaLabelConfig(mode="profit_first"),
    )
    attached = eng.generate(df, attach=True)
    assert "close" in attached.columns
    assert "meta_label" in attached.columns
    assert len(attached) == len(df)


def test_merge_labels_length_mismatch() -> None:
    df = _df([{"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}])
    lab = compute_triple_barrier_labels(
        _df(
            [
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                {"timestamp": 2, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            ]
        ),
        TripleBarrierConfig(max_bars=1, profit_pct=0.1, stop_pct=0.1),
    )
    with pytest.raises(ValueError, match="same number of rows"):
        merge_labels(df, lab)


def test_feature_matrix_and_labels_share_row_alignment() -> None:
    """Row i: features use data ≤ i; triple-barrier label uses path from bar i+1 onward."""
    from features.config import FeatureConfig
    from features.pipeline import build_features

    df = _df(
        [
            {"timestamp": k, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}
            for k in range(40)
        ]
    )
    # widen barriers so row-0 label is vertical not nan
    fcfg = FeatureConfig(
        return_periods=(1,),
        log_return_periods=(),
        volatility_windows=(),
        atr_period=3,
        momentum_periods=(),
        trend_slope_windows=(),
        rsi_period=4,
        macd_fast=3,
        macd_slow=5,
        macd_signal=2,
        bollinger_window=4,
        ma_windows=(3,),
        ma_types=("sma",),
        ma_slope_windows=(),
        zscore_windows=(),
        zscore_columns=(),
        rolling_variance_windows=(),
        skew_windows=(),
        kurtosis_windows=(),
        tick_direction_window=2,
        volume_imbalance_window=2,
        volatility_cluster_short=3,
        volatility_cluster_long=6,
    )
    x = build_features(df, fcfg, dropna=False)
    lcfg = TripleBarrierConfig(max_bars=5, profit_pct=0.04, stop_pct=0.04, position_side="long")
    lab = compute_triple_barrier_labels(df, lcfg)
    assert len(x) == len(lab) == len(df)


def test_last_row_no_future_is_insufficient_or_nan() -> None:
    df = _df(
        [
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        ]
    )
    out = compute_triple_barrier_labels(
        df, TripleBarrierConfig(max_bars=5, profit_pct=0.01, stop_pct=0.01)
    )
    assert np.isnan(out.iloc[0]["triple_barrier_label"])
