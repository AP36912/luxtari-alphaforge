"""Research markdown / bundle helpers."""

from __future__ import annotations

from pathlib import Path

from reporting.experiment_report import write_experiment_markdown


def test_write_experiment_markdown_contains_core_sections(tmp_path: Path) -> None:
    out = tmp_path / "REPORT.md"
    write_experiment_markdown(
        out,
        run_id="exp_unit_001",
        metrics={
            "config_hash": "abc",
            "seed": 1,
            "n_rows_used": 100,
            "label_horizon": 5,
            "cv": {
                "n_folds": 2,
                "accuracy_mean": 0.5,
                "precision_macro_mean": 0.4,
                "recall_macro_mean": 0.45,
            },
            "holdout_classification": {
                "n_train": 80,
                "n_test": 20,
                "accuracy": 0.55,
                "precision_macro": 0.5,
                "recall_macro": 0.5,
                "roc_auc_ovo": 0.6,
            },
            "holdout_backtest": {
                "sharpe": 0.1,
                "sortino": 0.2,
                "max_drawdown": -0.05,
                "win_rate": 0.51,
                "profit_factor": 1.1,
                "total_return": 0.02,
                "n_trades": 12,
            },
        },
        metadata={"asset": "TEST", "timeframe": "1d", "bar_type": "time"},
        label_spec={"mode": "triple_barrier", "max_bars": 5, "profit_pct": 0.02},
        model_name="random_forest",
        feature_list=["f1", "f2"],
        label_distribution={"-1.0": 10, "0.0": 5, "1.0": 12},
        resolved_data_path="/tmp/data.csv",
        audit_findings=[{"severity": "info", "code": "TS_MONOTONIC", "message": "ok"}],
    )
    text = out.read_text(encoding="utf-8")
    assert "exp_unit_001" in text
    assert "TEST" in text and "random_forest" in text
    assert "Top risks / caveats" in text
    assert "Purged K-Fold" in text
