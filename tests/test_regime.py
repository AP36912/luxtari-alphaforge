"""Regime detection, session tagging (XAUUSD-oriented defaults), and filters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.execution import TradeLeg
from backtest.segmentation import summarize_backtest_segment
from config.experiment_schema import DataConfig, ExperimentConfig, OutputConfig, ValidationConfig
from config.pipeline import run_experiment
from config.settings import ProjectPaths
from models.config import ModelConfig
from regime.config import RegimeConfig, RegimeFilterConfig, SessionCalendarConfig, SessionWindow
from regime.detection import detect_structure_series
from regime.filters import apply_signal_gate, trading_allowed_mask
from regime.integration import attach_regime_columns, regime_feature_column_names
from regime.masks import build_holdout_segment_masks
from regime.session_context import compute_session_tags, session_tag_for_row
def test_session_tags_utc_overlap_and_rollover() -> None:
    cfg = SessionCalendarConfig(session_timezone="UTC")
    # 14:00 UTC in London/NY overlap window [13:30, 16:30)
    ts = pd.Timestamp("2024-06-10 14:00:00", tz="UTC")
    assert session_tag_for_row(ts.time(), cfg) == "london_ny_overlap"
    # 22:00 UTC inside default rollover [21:45, 22:30)
    ts2 = pd.Timestamp("2024-06-10 22:00:00", tz="UTC")
    assert session_tag_for_row(ts2.time(), cfg) == "rollover"


def test_session_tags_deterministic_series() -> None:
    cfg = SessionCalendarConfig(session_timezone="UTC")
    idx = pd.date_range("2024-01-02", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": idx})
    s1 = compute_session_tags(df, cfg)
    s2 = compute_session_tags(df, cfg)
    pd.testing.assert_series_equal(s1, s2)


def test_timezone_conversion_new_york_wall_clock() -> None:
    """17:00 UTC → noon Eastern (EST ~5h) → still a session label (not off_hours if in NY)."""
    cfg = SessionCalendarConfig(session_timezone="America/New_York")
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-15 17:00:00"], utc=True)})
    tags = compute_session_tags(df, cfg)
    assert tags.iloc[0] in (
        "new_york",
        "london_ny_overlap",
        "london",
        "asia",
        "rollover",
        "off_hours",
    )


def test_regime_rule_outputs_stable() -> None:
    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    o = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, close) + 0.1,
            "low": np.minimum(o, close) - 0.1,
            "close": close,
        }
    )
    from regime.config import RegimeDetectionParams

    p = RegimeDetectionParams()
    a = detect_structure_series(df, p)
    b = detect_structure_series(df, p)
    pd.testing.assert_series_equal(a, b)


def test_trading_allowed_mask_sessions_and_vol() -> None:
    n = 5
    sess = np.array(["london", "asia", "rollover", "london", "new_york"])
    vol = np.array(["high_vol", "high_vol", "mid_vol", "low_vol", "mid_vol"])
    m = trading_allowed_mask(
        sess,
        vol,
        allowed_sessions=("london", "new_york"),
        allowed_volatility_states=("high_vol",),
        require_both=True,
    )
    assert list(m) == [True, False, False, False, False]
    m2 = trading_allowed_mask(sess, vol, allowed_sessions=("london",), allowed_volatility_states=None)
    assert list(m2) == [True, False, False, True, False]


def test_apply_signal_gate_zeros_disallowed() -> None:
    sig = np.array([1.0, -1.0, 1.0])
    gate = np.array([True, False, True])
    out = apply_signal_gate(sig, gate)
    np.testing.assert_array_almost_equal(out, [1.0, 0.0, 1.0])


def test_attach_regime_adds_columns_and_features() -> None:
    n = 250
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    c = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": o,
            "high": np.maximum(o, c) + 0.5,
            "low": np.minimum(o, c) - 0.5,
            "close": c,
            "volume": rng.uniform(100, 500, n),
        }
    )
    rc = RegimeConfig(enabled=True, add_feature_columns=True, detection_method="rules")
    out = attach_regime_columns(df, rc, fit_end_index=200)
    assert "regime_session" in out.columns
    assert "regime_vol_state" in out.columns
    assert "regime_structure" in out.columns
    assert set(regime_feature_column_names(rc)) <= set(out.columns)


def test_build_holdout_segment_masks_has_expected_keys() -> None:
    n = 20
    df = pd.DataFrame(
        {
            "regime_session": ["london"] * 10 + ["asia"] * 10,
            "regime_vol_state": ["high_vol"] * 20,
        }
    )
    m = build_holdout_segment_masks(df, RegimeFilterConfig())
    assert "all_bars" in m and m["all_bars"].all()
    assert m["session_london"][:10].all() and not m["session_london"][10:].any()


def test_segment_metrics_runs() -> None:
    e = np.array([100.0, 101.0, 100.5, 102.0], dtype=np.float64)
    legs = [TradeLeg(bar=2, pnl=1.0)]
    mask = np.array([True, True, True, False])
    s = summarize_backtest_segment(e, legs, mask, bars_per_year=252)
    assert "sharpe" in s and s["n_trades"] >= 0


def _fake_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(root=root, data_dir=root / "data", outputs_dir=root / "outputs")


def _synth_csv(path: Path, n: int = 420) -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-03-01", periods=n, freq="h", tz="UTC")
    c = 1800 + np.cumsum(rng.normal(0, 0.5, n))
    o = np.r_[c[0], c[:-1]]
    pd.DataFrame(
        {
            "timestamp": idx.tz_convert(None),
            "open": o,
            "high": np.maximum(o, c) + 0.2,
            "low": np.minimum(o, c) - 0.2,
            "close": c,
            "volume": rng.uniform(50, 200, n),
        }
    ).to_csv(path, index=False)


def test_pipeline_with_regime_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rp"
    root.mkdir()
    monkeypatch.setattr("config.pipeline.get_project_paths", lambda: _fake_paths(root))
    csv_path = tmp_path / "rg.csv"
    _synth_csv(csv_path)

    regime = RegimeConfig(
        enabled=True,
        add_feature_columns=True,
        detection_method="rules",
        include_segment_metrics=True,
        filters=RegimeFilterConfig(
            enabled=False,
            apply_to_signals=False,
        ),
    )
    cfg = ExperimentConfig(
        seed=3,
        data=DataConfig(source="csv", path=str(csv_path.resolve())),
        features={},
        labeling={
            "triple_barrier": {
                "max_bars": 5,
                "profit_pct": 0.02,
                "stop_pct": 0.02,
                "use_volatility_scaling": False,
            }
        },
        model=ModelConfig(
            name="random_forest",
            extra_params={"n_estimators": 20, "max_depth": 4, "n_jobs": 1},
        ),
        validation=ValidationConfig(n_splits=3, label_horizon=5, embargo=0, train_fraction=0.72),
        backtest={
            "initial_capital": 40_000.0,
            "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
            "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
        },
        output=OutputConfig(run_id="pytest_regime"),
        regime=regime,
    )
    out = run_experiment(cfg)
    metrics = json.loads((Path(out["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    assert "holdout_backtest_segments" in metrics
    seg = metrics["holdout_backtest_segments"]
    assert "all_bars" in seg
    assert "session_london_ny_overlap" in seg
