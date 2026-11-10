"""
Rolling / expanding walk-forward: train on purge-safe train index, backtest on test segment.

Uses :class:`validation.walk_forward.WalkForwardSplitter` for indices. Reuses the same
feature/label/backtest stack as :func:`config.pipeline.run_experiment` without altering
model, feature, or execution implementations.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.metrics import summarize
from config.experiment_schema import (
    ExperimentConfig,
    backtest_config_from_dict,
    feature_config_from_overrides,
    triple_barrier_config_from_dict,
)
from config.pipeline import load_ohlcv_frame, set_global_seeds
from config.settings import get_project_paths
from features.pipeline import build_features, feature_column_names
from labeling.triple_barrier import compute_triple_barrier_labels, merge_labels
from models.config import ModelConfig
from models.registry import create_model
from models.training import ModelTrainer
from evaluation.probability_calibration import (
    calibration_holdout_metrics,
    fit_prefit_calibrator,
    time_ordered_fit_cal_split,
)
from evaluation.signal_diagnostics import (
    finalize_weighted_signal_stats,
    merge_weighted_signal_stats,
    prediction_probability_stats,
)
from evaluation.walk_forward_entry_gate import (
    apply_walk_forward_entry_regime_blocks,
    trades_by_entry_vol_regime,
)
from evaluation.walk_forward_exit_rule import apply_walk_forward_cohort_exit_rule
from regime.filters import apply_signal_gate, trading_allowed_mask
from regime.integration import attach_regime_columns, regime_feature_column_names
from validation.walk_forward import WalkForwardSplitter

logger = logging.getLogger(__name__)

WindowMode = Literal["expanding", "rolling"]


def _unwrap_sklearn_estimator(model: Any) -> Any | None:
    """``RandomForestModel`` / boosting wrappers expose a fitted ``_estimator``."""
    est = getattr(model, "_estimator", None)
    if est is None or not hasattr(est, "predict_proba"):
        return None
    return est


def _aggregate_calibration_holdout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted by ``n_cal`` rows per window (calibration tail of train only)."""
    if not rows:
        return {}
    w = float(sum(int(r["n_cal"]) for r in rows))
    if w <= 0:
        return {}

    def wavg(key: str) -> float:
        return sum(float(r[key]) * int(r["n_cal"]) for r in rows) / w

    bu, bc = wavg("brier_multiclass_uncalibrated"), wavg("brier_multiclass_calibrated")
    eu, ec = wavg("top_class_ece_uncalibrated"), wavg("top_class_ece_calibrated")
    return {
        "n_windows_with_calibration_fit": len(rows),
        "holdout_weighted_mean_brier_uncalibrated": round(bu, 8),
        "holdout_weighted_mean_brier_calibrated": round(bc, 8),
        "holdout_weighted_mean_delta_brier_uncal_minus_cal": round(bu - bc, 8),
        "holdout_weighted_mean_top_class_ece_uncalibrated": round(eu, 8),
        "holdout_weighted_mean_top_class_ece_calibrated": round(ec, 8),
    }


def _seed_model_params(extra: dict[str, Any], seed: int) -> dict[str, Any]:
    out = dict(extra)
    out["random_state"] = int(seed)
    return out


def _labeling_horizon_and_spec(cfg: ExperimentConfig) -> tuple[int, str, dict[str, Any]]:
    """Public edition supports triple-barrier labeling only."""
    labeling_cfg = dict(cfg.labeling or {})
    label_mode = str(labeling_cfg.get("mode") or "triple_barrier")
    if label_mode not in {"triple_barrier", "tb", ""}:
        raise ValueError(
            f"unsupported labeling.mode={label_mode!r}; public edition supports triple_barrier only"
        )
    tb = triple_barrier_config_from_dict(labeling_cfg)
    horizon = int(tb.max_bars)
    label_spec = {**dataclasses.asdict(tb), "mode": "triple_barrier"}
    return horizon, "triple_barrier", label_spec


def _build_merged_frame(
    cfg: ExperimentConfig,
    *,
    ohlcv_row_slice: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], int, str]:
    """
    Load CSV → features → labels → merged matrix rows (same as experiment pipeline).

    ``ohlcv_row_slice`` is an optional half-open ``[start, end)`` range of **raw OHLCV**
    rows as loaded from CSV (before feature ``dropna``), matching robustness sweep
    ``center_shift_*`` semantics on the same file.
    """
    horizon, label_mode, _label_spec = _labeling_horizon_and_spec(cfg)
    raw = load_ohlcv_frame(cfg)
    if ohlcv_row_slice is not None:
        lo, hi = int(ohlcv_row_slice[0]), int(ohlcv_row_slice[1])
        n_full = len(raw)
        if lo < 0 or hi > n_full or lo >= hi:
            raise ValueError(
                f"ohlcv_row_slice [{lo}, {hi}) invalid for OHLCV frame with {n_full} rows"
            )
        raw = raw.iloc[lo:hi].reset_index(drop=True)
    feat_cfg = feature_config_from_overrides(cfg.features)
    featured = build_features(raw, feat_cfg, dropna=True, include_ohlcv=True)
    tb = triple_barrier_config_from_dict(dict(cfg.labeling or {}))
    lab_df = compute_triple_barrier_labels(featured, tb)
    merged = merge_labels(featured, lab_df)
    y = merged["triple_barrier_label"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    merged = merged.loc[valid].reset_index(drop=True)
    y = merged["triple_barrier_label"].to_numpy(dtype=np.int64)
    feat_names_base = feature_column_names(featured)
    if not feat_names_base:
        raise ValueError("No feature columns after pipeline; check FeatureConfig")
    return merged, y, feat_names_base, horizon, label_mode


@dataclass
class WalkForwardParams:
    """CLI-facing walk-forward splitter parameters."""

    test_size: int
    initial_train: int
    window: WindowMode = "expanding"
    train_size: int | None = None
    step: int | None = None
    embargo: int | None = None
    calibration_method: Literal["none", "sigmoid", "isotonic"] = "none"
    """Platt (``sigmoid``) or ``isotonic`` on a time-ordered tail of **train only**; test never used."""
    calibration_holdout_fraction: float = 0.15
    calibration_min_cal_rows: int = 50
    calibration_min_fit_rows: int = 50
    eval_ohlcv_rows: tuple[int, int] | None = None
    """If set, use only ``raw.iloc[start:end]`` (half-open) before features — raw CSV indices."""
    eval_merged_timestamp_inclusive: tuple[pd.Timestamp, pd.Timestamp] | None = None
    """If set, keep only merged rows with ``timestamp`` in ``[t0, t1]`` (inclusive)."""


def run_walk_forward_evaluation(
    cfg: ExperimentConfig,
    wf: WalkForwardParams,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    For each walk-forward split: fit a fresh model on train rows, backtest on test OHLC
    with classifier signals (and optional regime signal gate).

    Returns a dict with ``windows`` rows, ``aggregate`` summary, and written paths.
    """
    paths = get_project_paths()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = output_dir or (paths.root / "outputs" / "reports" / "walk_forward" / f"walk_forward_{stamp}")
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    set_global_seeds(cfg.seed)
    embargo = int(wf.embargo if wf.embargo is not None else cfg.validation.embargo)

    merged, y, feat_names_base, horizon, label_mode = _build_merged_frame(
        cfg, ohlcv_row_slice=wf.eval_ohlcv_rows
    )
    if wf.eval_merged_timestamp_inclusive is not None:
        t0, t1 = wf.eval_merged_timestamp_inclusive
        ts = merged["timestamp"]
        merged = merged.loc[(ts >= t0) & (ts <= t1)].reset_index(drop=True)
        y = merged["triple_barrier_label"].to_numpy(dtype=np.int64)
    n = len(merged)
    r_cfg = cfg.regime

    splitter = WalkForwardSplitter(
        wf.test_size,
        train_size=wf.train_size,
        window=wf.window,
        step=wf.step,
        embargo=embargo,
        label_horizon=horizon,
        initial_train=wf.initial_train,
    )

    X_dummy = np.zeros((n, 1), dtype=np.float64)
    splits = list(splitter.split(X_dummy))
    if not splits:
        raise ValueError(
            f"No walk-forward splits (n={n}). Relax test_size / initial_train / embargo "
            f"or use more data."
        )

    bt_cfg: BacktestConfig = backtest_config_from_dict(cfg.backtest)

    window_rows: list[dict[str, Any]] = []
    equity_chain: list[float] = [1.0]
    pf_values: list[float] = []
    returns: list[float] = []
    signal_prob_acc: dict[str, float] = {}
    signal_prob_acc_uncal_test: dict[str, float] = {}
    calibration_holdout_rows: list[dict[str, Any]] = []

    for i, (tr_idx, te_idx) in enumerate(splits):
        fit_end = int(np.max(tr_idx)) + 1
        merged_i = (
            attach_regime_columns(merged, r_cfg, fit_end_index=fit_end)
            if r_cfg.enabled
            else merged
        )
        feat_names = feat_names_base + regime_feature_column_names(r_cfg)
        X_i = merged_i[feat_names].to_numpy(dtype=np.float64)

        y_tr = y[tr_idx]
        if len(np.unique(y_tr)) < 2:
            logger.warning("walk_forward window %s: skip — single class in train", i)
            window_rows.append(
                {
                    "window_index": i,
                    "skipped": True,
                    "skip_reason": "single_class_train",
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                    "train_start_row": int(tr_idx.min()),
                    "train_end_row": int(tr_idx.max()),
                    "test_start_row": int(te_idx.min()),
                    "test_end_row": int(te_idx.max()),
                }
            )
            continue

        mconf = ModelConfig(
            name=cfg.model.name,
            extra_params=_seed_model_params(dict(cfg.model.extra_params), cfg.seed + i),
        )
        trainer = ModelTrainer(create_model(mconf))
        pred_te: np.ndarray
        sig_prob_diag: dict[str, Any] | None = None
        sig_prob_diag_uncal_test: dict[str, Any] | None = None
        prob_cal: dict[str, Any] = {"applied": False, "method": wf.calibration_method}

        cm = wf.calibration_method
        if cm in ("sigmoid", "isotonic"):
            split = time_ordered_fit_cal_split(
                tr_idx,
                y,
                holdout_fraction=float(wf.calibration_holdout_fraction),
                min_cal_rows=int(wf.calibration_min_cal_rows),
                min_fit_rows=int(wf.calibration_min_fit_rows),
            )
            base_est = None
            if split is not None:
                fit_i, cal_i = split
                trainer.fit(X_i[fit_i], y[fit_i])
                base_est = _unwrap_sklearn_estimator(trainer.model)
            if split is None or base_est is None:
                logger.warning(
                    "walk_forward window %s: skip calibration (valid_split=%s, base_est=%s); "
                    "uncalibrated full train",
                    i,
                    split is not None,
                    base_est is not None,
                )
                trainer = ModelTrainer(create_model(mconf))
                trainer.fit(X_i[tr_idx], y_tr)
                pred_te = trainer.predict(X_i[te_idx])
                prob_cal = {
                    "applied": False,
                    "method": cm,
                    "reason": "split_or_non_sklearn_estimator",
                }
                try:
                    proba_te = trainer.predict_proba(X_i[te_idx])
                    sig_prob_diag = prediction_probability_stats(proba_te)
                    merge_weighted_signal_stats(signal_prob_acc, sig_prob_diag, int(len(te_idx)))
                except Exception:
                    logger.debug("predict_proba unavailable for window %s", i, exc_info=True)
            else:
                fit_i, cal_i = split
                X_cal, y_cal = X_i[cal_i], y[cal_i]
                X_te = X_i[te_idx]
                try:
                    cal_wrapped = fit_prefit_calibrator(base_est, X_cal, y_cal, method=cm)
                    cls_ord = np.asarray(cal_wrapped.classes_)
                    met = calibration_holdout_metrics(
                        base_est, cal_wrapped, X_cal, y_cal, classes=cls_ord
                    )
                    pred_te = np.asarray(cal_wrapped.predict(X_te), dtype=np.float64)
                    proba_te_cal = cal_wrapped.predict_proba(X_te)
                    proba_te_uncal = np.asarray(base_est.predict_proba(X_te), dtype=np.float64)
                    sig_prob_diag = prediction_probability_stats(proba_te_cal)
                    sig_prob_diag_uncal_test = prediction_probability_stats(proba_te_uncal)
                    merge_weighted_signal_stats(signal_prob_acc, sig_prob_diag, int(len(te_idx)))
                    merge_weighted_signal_stats(
                        signal_prob_acc_uncal_test, sig_prob_diag_uncal_test, int(len(te_idx))
                    )
                    delta_b = float(met["brier_multiclass_uncalibrated"]) - float(
                        met["brier_multiclass_calibrated"]
                    )
                    prob_cal = {
                        "applied": True,
                        "method": cm,
                        "n_train_fit": int(len(fit_i)),
                        "n_train_cal": int(len(cal_i)),
                        "brier_multiclass_uncalibrated": met["brier_multiclass_uncalibrated"],
                        "brier_multiclass_calibrated": met["brier_multiclass_calibrated"],
                        "delta_brier_uncal_minus_cal": round(delta_b, 8),
                        "top_class_ece_uncalibrated": met["top_class_ece_uncalibrated"],
                        "top_class_ece_calibrated": met["top_class_ece_calibrated"],
                    }
                    calibration_holdout_rows.append({**met, "n_cal": int(len(cal_i))})
                except Exception as exc:
                    logger.warning("walk_forward window %s: calibration fit failed %s", i, exc)
                    trainer = ModelTrainer(create_model(mconf))
                    trainer.fit(X_i[tr_idx], y_tr)
                    pred_te = trainer.predict(X_te)
                    prob_cal = {"applied": False, "method": cm, "reason": f"fit_error:{exc}"}
                    try:
                        proba_te = trainer.predict_proba(X_te)
                        sig_prob_diag = prediction_probability_stats(proba_te)
                        merge_weighted_signal_stats(signal_prob_acc, sig_prob_diag, int(len(te_idx)))
                    except Exception:
                        logger.debug("predict_proba unavailable for window %s", i, exc_info=True)
        else:
            trainer.fit(X_i[tr_idx], y_tr)
            pred_te = trainer.predict(X_i[te_idx])
            try:
                proba_te = trainer.predict_proba(X_i[te_idx])
                sig_prob_diag = prediction_probability_stats(proba_te)
                merge_weighted_signal_stats(signal_prob_acc, sig_prob_diag, int(len(te_idx)))
            except Exception:
                logger.debug("predict_proba unavailable for window %s", i, exc_info=True)

        merged_te = merged_i.iloc[te_idx].reset_index(drop=True)
        signals = np.asarray(pred_te, dtype=np.float64)
        entry_gate_diag: dict[str, Any] = {}
        exit_rule_diag: dict[str, Any] = {}
        trades_vol: dict[str, int] = {}
        if r_cfg.enabled and r_cfg.filters.enabled and r_cfg.filters.apply_to_signals:
            sess_te = merged_te["regime_session"].values
            vol_te = merged_te["regime_vol_state"].values
            struct_te = merged_te["regime_structure"].values
            use_vol_in_mask = bool(r_cfg.filters.allowed_volatility_states) and not (
                r_cfg.filters.block_high_vol_entries or r_cfg.filters.block_trending_down_entries
            )
            gate = trading_allowed_mask(
                sess_te,
                vol_te,
                allowed_sessions=r_cfg.filters.allowed_sessions if r_cfg.filters.allowed_sessions else None,
                allowed_volatility_states=r_cfg.filters.allowed_volatility_states
                if use_vol_in_mask
                else None,
                require_both=r_cfg.filters.require_both,
            )
            signals = apply_signal_gate(signals, gate)
            _any_entry_block = (
                r_cfg.filters.block_high_vol_entries
                or r_cfg.filters.block_trending_down_entries
                or r_cfg.filters.block_lowvol_rangebound_london
            )
            if _any_entry_block:
                signals, entry_gate_diag = apply_walk_forward_entry_regime_blocks(
                    signals,
                    vol_te,
                    struct_te,
                    fill_basis=str(bt_cfg.assumptions.fill_basis),
                    sizing=bt_cfg.sizing,
                    block_high_vol=bool(r_cfg.filters.block_high_vol_entries),
                    block_trending_down=bool(r_cfg.filters.block_trending_down_entries),
                    block_lowvol_rb_london=bool(r_cfg.filters.block_lowvol_rangebound_london),
                    session_states=sess_te,
                )

        ohlc_te = merged_te[["open", "high", "low", "close"]]
        if r_cfg.enabled and r_cfg.filters.enabled and r_cfg.filters.apply_to_signals:
            vol_te = merged_te["regime_vol_state"].values
            struct_te = merged_te["regime_structure"].values
            signals, exit_rule_diag = apply_walk_forward_cohort_exit_rule(
                signals,
                ohlc_te,
                vol_te,
                struct_te,
                filters=r_cfg.filters,
                fill_basis=str(bt_cfg.assumptions.fill_basis),
            )
        engine = BacktestEngine(bt_cfg)
        bt_result = engine.run(ohlc_te, signals)
        if r_cfg.enabled and "regime_vol_state" in merged_te.columns:
            trades_vol = trades_by_entry_vol_regime(
                bt_result.position,
                merged_te["regime_vol_state"].to_numpy(),
            )
        bt_m = summarize(
            bt_result.equity,
            bt_result.trade_legs,
            bars_per_year=bt_cfg.bars_per_year,
        )

        tr_ret = float(bt_m.total_return)
        returns.append(tr_ret)
        new_eq = equity_chain[-1] * (1.0 + tr_ret)
        equity_chain.append(new_eq)

        pf_raw = bt_m.profit_factor
        if np.isfinite(pf_raw):
            pf_values.append(float(pf_raw))

        ts_first = merged_te["timestamp"].iloc[0] if "timestamp" in merged_te.columns else None
        ts_last = merged_te["timestamp"].iloc[-1] if "timestamp" in merged_te.columns else None

        window_rows.append(
            {
                "window_index": i,
                "skipped": False,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "train_start_row": int(tr_idx.min()),
                "train_end_row": int(tr_idx.max()),
                "test_start_row": int(te_idx.min()),
                "test_end_row": int(te_idx.max()),
                "test_first_timestamp": str(ts_first) if ts_first is not None else None,
                "test_last_timestamp": str(ts_last) if ts_last is not None else None,
                "sharpe": bt_m.sharpe,
                "sortino": bt_m.sortino,
                "max_drawdown": bt_m.max_drawdown,
                "win_rate": bt_m.win_rate,
                "profit_factor": None if np.isinf(pf_raw) or np.isnan(pf_raw) else float(pf_raw),
                "profit_factor_infinite": bool(np.isinf(pf_raw)),
                "total_return": tr_ret,
                "n_trades": bt_m.n_trades,
                "equity_end_normalized": new_eq,
                "walk_forward_entry_gate": entry_gate_diag,
                "walk_forward_exit_rule": exit_rule_diag,
                "trades_by_entry_vol_regime": trades_vol,
                "signal_probability_diagnostics": sig_prob_diag,
                "signal_probability_diagnostics_test_uncalibrated": sig_prob_diag_uncal_test,
                "probability_calibration": prob_cal,
            }
        )

    ran = [w for w in window_rows if not w.get("skipped")]
    n_ran = len(ran)
    pf_mean = float(np.mean(pf_values)) if pf_values else None
    ret_arr = np.asarray(returns, dtype=np.float64)

    blocked_hv_total = 0
    entry_opp_total = 0
    forced_exit_total = 0
    trades_vol_total: dict[str, int] = {}
    for w in ran:
        g = w.get("walk_forward_entry_gate") or {}
        ex = w.get("walk_forward_exit_rule") or {}
        blocked_hv_total += int(g.get("entries_blocked_high_vol") or 0)
        entry_opp_total += int(g.get("n_flat_to_open_opportunities") or 0)
        forced_exit_total += int(ex.get("forced_exits") or 0)
        for k, v in (w.get("trades_by_entry_vol_regime") or {}).items():
            trades_vol_total[k] = trades_vol_total.get(k, 0) + int(v)
    pct_hv_weighted = (
        100.0 * blocked_hv_total / entry_opp_total if entry_opp_total else None
    )

    aggregate: dict[str, Any] = {
        "n_splits_configured": len(splits),
        "n_windows_ran": n_ran,
        "n_windows_skipped": len(window_rows) - n_ran,
        "mean_profit_factor_finite_windows": pf_mean,
        "n_windows_with_finite_profit_factor": len(pf_values),
        "mean_total_return_per_window": float(np.mean(ret_arr)) if n_ran else None,
        "std_total_return_per_window": float(np.std(ret_arr, ddof=0)) if n_ran > 1 else 0.0,
        "windows_with_positive_return": int(np.sum(ret_arr > 0)) if n_ran else 0,
        "windows_with_negative_return": int(np.sum(ret_arr < 0)) if n_ran else 0,
        "equity_curve_normalized": equity_chain,
        "final_compounded_equity": equity_chain[-1] if equity_chain else 1.0,
        "stability_profit_factor_std": float(np.std(pf_values, ddof=0)) if len(pf_values) > 1 else 0.0,
        "entries_blocked_high_vol_total": int(blocked_hv_total),
        "n_flat_to_open_opportunities_total": int(entry_opp_total),
        "pct_entry_opportunities_blocked_high_vol": (
            round(float(pct_hv_weighted), 6) if pct_hv_weighted is not None else None
        ),
        "cohort_forced_exits_total": int(forced_exit_total),
        "trades_by_entry_vol_regime_all_windows": dict(sorted(trades_vol_total.items())),
        "signal_probability_diagnostics_weighted_mean": finalize_weighted_signal_stats(signal_prob_acc),
        "calibration_holdout_quality_weighted_mean": _aggregate_calibration_holdout_rows(
            calibration_holdout_rows
        ),
    }
    _uncal_wm = finalize_weighted_signal_stats(signal_prob_acc_uncal_test)
    if _uncal_wm:
        aggregate["signal_probability_diagnostics_test_uncalibrated_weighted_mean"] = _uncal_wm

    summary = {
        "generated_utc": stamp,
        "labeling_mode": label_mode,
        "label_horizon": horizon,
        "walk_forward": {
            "window": wf.window,
            "test_size": wf.test_size,
            "initial_train": wf.initial_train,
            "train_size": wf.train_size,
            "step": wf.step,
            "embargo": embargo,
            "calibration_method": wf.calibration_method,
            "calibration_holdout_fraction": wf.calibration_holdout_fraction,
            "calibration_min_cal_rows": wf.calibration_min_cal_rows,
            "calibration_min_fit_rows": wf.calibration_min_fit_rows,
            "eval_ohlcv_rows_half_open": (
                [int(wf.eval_ohlcv_rows[0]), int(wf.eval_ohlcv_rows[1])]
                if wf.eval_ohlcv_rows is not None
                else None
            ),
            "eval_merged_timestamp_inclusive": (
                [str(wf.eval_merged_timestamp_inclusive[0]), str(wf.eval_merged_timestamp_inclusive[1])]
                if wf.eval_merged_timestamp_inclusive is not None
                else None
            ),
        },
        "n_rows_labeled": n,
        "aggregate": aggregate,
        "windows": window_rows,
    }

    json_path = out_root / "walk_forward_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Walk-forward evaluation",
        "",
        f"- generated_utc: `{stamp}`",
        f"- output_dir: `{out_root}`",
        f"- labeling_mode: `{label_mode}`",
        f"- label_horizon: `{horizon}`",
        f"- window: `{wf.window}` | test_size={wf.test_size} | initial_train={wf.initial_train} "
        f"| train_size={wf.train_size} | step={wf.step} | embargo={embargo}",
        "",
        "## Aggregate",
        "",
        f"- windows ran: `{aggregate['n_windows_ran']}` (skipped: `{aggregate['n_windows_skipped']}`)",
        f"- mean profit factor (finite windows): `{aggregate['mean_profit_factor_finite_windows']}`",
        f"- mean total return / window: `{aggregate['mean_total_return_per_window']}`",
        f"- std total return / window: `{aggregate['std_total_return_per_window']}`",
        f"- positive-return windows: `{aggregate['windows_with_positive_return']}`",
        f"- negative-return windows: `{aggregate['windows_with_negative_return']}`",
        f"- std profit factor (finite): `{aggregate['stability_profit_factor_std']}`",
        f"- compounded equity (× from 1.0): `{aggregate['final_compounded_equity']}`",
        f"- entries_blocked_high_vol (total): `{aggregate.get('entries_blocked_high_vol_total')}`",
        f"- flat→open opportunities (total): `{aggregate.get('n_flat_to_open_opportunities_total')}`",
        f"- % entry opportunities blocked (high vol): `{aggregate.get('pct_entry_opportunities_blocked_high_vol')}`",
        f"- trades by entry vol regime (sum): `{aggregate.get('trades_by_entry_vol_regime_all_windows')}`",
        f"- signal prob (weighted mean test): `{aggregate.get('signal_probability_diagnostics_weighted_mean')}`",
        f"- signal prob test **uncalibrated** (only if calibration on): `{aggregate.get('signal_probability_diagnostics_test_uncalibrated_weighted_mean', {})}`",
        f"- calibration holdout quality (train tail, weighted): `{aggregate.get('calibration_holdout_quality_weighted_mean')}`",
        "",
        "## Per-window",
        "",
        "| i | train rows | test rows | trades | win_rate | PF | total_ret | eq× |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for w in window_rows:
        if w.get("skipped"):
            md_lines.append(
                f"| {w['window_index']} | {w.get('n_train')} | {w.get('n_test')} | — | — | SKIP | — | — |"
            )
        else:
            pf_s = "inf" if w.get("profit_factor_infinite") else str(w.get("profit_factor"))
            wr = w.get("win_rate")
            wr_s = f"{float(wr):.4f}" if wr is not None and np.isfinite(wr) else str(wr)
            md_lines.append(
                f"| {w['window_index']} | {w['n_train']} | {w['n_test']} | {w['n_trades']} | "
                f"{wr_s} | {pf_s} | {w['total_return']:.6f} | {w['equity_end_normalized']:.6f} |"
            )
    md_lines.append("")
    md_lines.append(f"Full JSON: `{json_path}`")
    md_path = out_root / "walk_forward_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Walk-forward wrote %s and %s", json_path, md_path)
    return {
        "output_dir": str(out_root),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "summary": summary,
    }


__all__ = ["run_walk_forward_evaluation", "WalkForwardParams"]
