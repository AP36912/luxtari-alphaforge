"""
Walk-forward trade attribution for diagnostics (read-only analysis).

Mirrors :func:`evaluation.walk_forward_runner.run_walk_forward_evaluation` per-window
training, calibration, and backtest, but returns per–round-trip trade rows and
weak-regime test-bar probability rows instead of writing summaries.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from config.experiment_schema import backtest_config_from_dict
from backtest.execution import TradeLeg
from config.experiment_schema import ExperimentConfig
from config.pipeline import set_global_seeds
from evaluation.probability_calibration import fit_prefit_calibrator, time_ordered_fit_cal_split
from evaluation.walk_forward_entry_gate import apply_walk_forward_entry_regime_blocks
from evaluation.walk_forward_exit_rule import apply_walk_forward_cohort_exit_rule
from evaluation.walk_forward_runner import (
    WalkForwardParams,
    _build_merged_frame,
    _seed_model_params,
    _unwrap_sklearn_estimator,
)
from models.config import ModelConfig
from models.registry import create_model
from models.training import ModelTrainer
from regime.filters import apply_signal_gate, trading_allowed_mask
from regime.integration import attach_regime_columns, regime_feature_column_names
from validation.walk_forward import WalkForwardSplitter

logger = logging.getLogger(__name__)


def _pnl_by_exit_bar(legs: list[TradeLeg]) -> dict[int, float]:
    acc: dict[int, float] = {}
    for lg in legs:
        acc[lg.bar] = acc.get(lg.bar, 0.0) + float(lg.pnl)
    return acc


def extract_round_trips_from_position(pos: np.ndarray, legs: list[TradeLeg]) -> list[dict[str, Any]]:
    """
    Reconstruct flat-to-flat (or reversal) segments from ``BacktestResult.position``.

    ``next_bar_open`` fill: entry at bar ``i`` uses signal at ``i-1`` (``entry_sig_bar``).
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    by_bar = _pnl_by_exit_bar(legs)
    out: list[dict[str, Any]] = []
    if len(pos) < 2:
        return out

    in_trade = False
    entry_exec = 0
    entry_sig = 0
    direction = 0

    def _dir(p: float) -> int:
        if p > 1e-12:
            return 1
        if p < -1e-12:
            return -1
        return 0

    for i in range(1, len(pos)):
        prev_d = _dir(float(pos[i - 1]))
        cur_d = _dir(float(pos[i]))
        prev_z = prev_d == 0
        cur_z = cur_d == 0

        if not in_trade and prev_z and not cur_z:
            in_trade = True
            entry_exec = i
            entry_sig = i - 1
            direction = cur_d
        elif in_trade:
            reversal = not cur_z and cur_d != 0 and cur_d != direction
            if cur_z or reversal:
                pnl = float(by_bar.get(i, 0.0))
                out.append(
                    {
                        "entry_exec_bar": int(entry_exec),
                        "entry_sig_bar": int(entry_sig),
                        "exit_exec_bar": int(i),
                        "hold_bars": int(i - entry_exec),
                        "direction": int(direction),
                        "pnl": pnl,
                    }
                )
                if cur_z:
                    in_trade = False
                else:
                    in_trade = True
                    entry_exec = i
                    entry_sig = i - 1
                    direction = cur_d
    return out


def compute_mae_mfe(
    ohlc: pd.DataFrame,
    entry_exec_bar: int,
    exit_exec_bar: int,
    direction: int,
    entry_price: float,
) -> dict[str, float | None]:
    """
    MAE/MFE in raw price and ATR-normalized terms for a single round trip.

    ``ohlc`` is the per-window test OHLC (columns: open, high, low, close).
    ``entry_price`` is ``open[entry_exec_bar]`` (next-bar-open fill assumption).
    """
    lo = max(0, entry_exec_bar)
    hi = min(len(ohlc), exit_exec_bar + 1)
    if hi <= lo or direction == 0 or entry_price <= 0:
        return {"mae_raw": None, "mfe_raw": None, "mae_atr": None, "mfe_atr": None}

    highs = ohlc["high"].iloc[lo:hi].to_numpy(dtype=np.float64)
    lows = ohlc["low"].iloc[lo:hi].to_numpy(dtype=np.float64)
    closes = ohlc["close"].iloc[lo:hi].to_numpy(dtype=np.float64)

    if direction > 0:
        worst = float(np.min(lows)) - entry_price
        best = float(np.max(highs)) - entry_price
    else:
        worst = entry_price - float(np.max(highs))
        best = entry_price - float(np.min(lows))

    mae = float(min(worst, 0.0))
    mfe = float(max(best, 0.0))

    # ATR-14 at entry bar (causal: uses data up to and including entry_exec_bar)
    atr_window = 14
    if entry_exec_bar >= atr_window:
        pre = max(0, entry_exec_bar - atr_window + 1)
        seg_h = ohlc["high"].iloc[pre : entry_exec_bar + 1].to_numpy(dtype=np.float64)
        seg_l = ohlc["low"].iloc[pre : entry_exec_bar + 1].to_numpy(dtype=np.float64)
        seg_c = ohlc["close"].iloc[pre : entry_exec_bar + 1].to_numpy(dtype=np.float64)
        prev_c = np.concatenate([[np.nan], seg_c[:-1]])
        tr = np.nanmax(
            np.column_stack([seg_h - seg_l, np.abs(seg_h - prev_c), np.abs(seg_l - prev_c)]),
            axis=1,
        )
        atr_val = float(np.nanmean(tr[1:]))  # skip first NaN from prev_c
    else:
        atr_val = 0.0

    if atr_val > 1e-9:
        mae_atr = mae / atr_val
        mfe_atr = mfe / atr_val
    else:
        mae_atr = None
        mfe_atr = None

    return {"mae_raw": mae, "mfe_raw": mfe, "mae_atr": mae_atr, "mfe_atr": mfe_atr}


def _confidence_bucket(max_p: float) -> str:
    if max_p < 0.40:
        return "pmax_lt_0.40"
    if max_p < 0.55:
        return "pmax_0.40_0.55"
    return "pmax_ge_0.55"


def collect_walk_forward_trade_attribution(
    cfg: ExperimentConfig,
    wf: WalkForwardParams,
    *,
    slice_label: str,
) -> dict[str, Any]:
    """
    Run walk-forward on the merged frame implied by ``wf`` (timestamp band, etc.),
    returning trade rows and per–test-bar rows for ``mid_vol`` + ``range_bound``.
    """
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
        raise ValueError("No walk-forward splits")

    bt_cfg = backtest_config_from_dict(cfg.backtest)
    trade_rows: list[dict[str, Any]] = []
    weak_bar_rows: list[dict[str, Any]] = []

    for wi, (tr_idx, te_idx) in enumerate(splits):
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
            continue

        mconf = ModelConfig(
            name=cfg.model.name,
            extra_params=_seed_model_params(dict(cfg.model.extra_params), cfg.seed + wi),
        )
        trainer = ModelTrainer(create_model(mconf))
        pred_te: np.ndarray
        proba_te: np.ndarray | None = None

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
                trainer = ModelTrainer(create_model(mconf))
                trainer.fit(X_i[tr_idx], y_tr)
                pred_te = trainer.predict(X_i[te_idx])
                try:
                    proba_te = trainer.predict_proba(X_i[te_idx])
                except Exception:
                    proba_te = None
            else:
                fit_i, cal_i = split
                X_cal, y_cal = X_i[cal_i], y[cal_i]
                X_te = X_i[te_idx]
                try:
                    cal_wrapped = fit_prefit_calibrator(base_est, X_cal, y_cal, method=cm)
                    pred_te = np.asarray(cal_wrapped.predict(X_te), dtype=np.float64)
                    proba_te = cal_wrapped.predict_proba(X_te)
                except Exception as exc:
                    logger.warning("calibration failed window %s: %s", wi, exc)
                    trainer = ModelTrainer(create_model(mconf))
                    trainer.fit(X_i[tr_idx], y_tr)
                    pred_te = trainer.predict(X_te)
                    try:
                        proba_te = trainer.predict_proba(X_te)
                    except Exception:
                        proba_te = None
        else:
            trainer.fit(X_i[tr_idx], y_tr)
            pred_te = trainer.predict(X_i[te_idx])
            try:
                proba_te = trainer.predict_proba(X_i[te_idx])
            except Exception:
                proba_te = None

        merged_te = merged_i.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]
        signals = np.asarray(pred_te, dtype=np.float64)

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
                signals, _ = apply_walk_forward_entry_regime_blocks(
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
            signals, _ = apply_walk_forward_cohort_exit_rule(
                signals,
                ohlc_te,
                vol_te,
                struct_te,
                filters=r_cfg.filters,
                fill_basis=str(bt_cfg.assumptions.fill_basis),
            )
        engine = BacktestEngine(bt_cfg)
        bt_result = engine.run(ohlc_te, signals)
        trips = extract_round_trips_from_position(bt_result.position, bt_result.trade_legs)

        vol_arr = merged_te["regime_vol_state"].astype(str).values
        struct_arr = merged_te["regime_structure"].astype(str).values
        sess_arr = (
            merged_te["regime_session"].astype(str).values
            if "regime_session" in merged_te.columns
            else None
        )

        if proba_te is not None and proba_te.ndim == 2:
            max_p = proba_te.max(axis=1)
            sp = np.sort(proba_te, axis=1)
            margin = sp[:, -1] - sp[:, -2]
        else:
            max_p = np.full(len(merged_te), np.nan)
            margin = np.full(len(merged_te), np.nan)

        for t in trips:
            es = int(t["entry_sig_bar"])
            if es < 0 or es >= len(merged_te):
                continue
            vol_e = str(vol_arr[es])
            struct_e = str(struct_arr[es])
            sess_e = str(sess_arr[es]) if sess_arr is not None else ""
            pred_e = float(pred_te[es])
            y_e = int(y_te[es])
            dir_e = int(np.sign(pred_e)) if abs(pred_e) > 1e-12 else 0
            mp = float(max_p[es]) if np.isfinite(max_p[es]) else None
            mg = float(margin[es]) if np.isfinite(margin[es]) else None
            entry_exec = int(t["entry_exec_bar"])
            exit_exec = int(t["exit_exec_bar"])
            entry_px = float(ohlc_te["open"].iloc[entry_exec]) if entry_exec < len(ohlc_te) else 0.0
            excursion = compute_mae_mfe(ohlc_te, entry_exec, exit_exec, dir_e, entry_px)
            trade_rows.append(
                {
                    "slice": slice_label,
                    "window_index": int(wi),
                    "volatility_regime": vol_e,
                    "regime_structure": struct_e,
                    "entry_session": sess_e,
                    "direction": dir_e,
                    "pred_class": int(round(pred_e)),
                    "label_at_signal_bar": y_e,
                    "class_match": bool(int(round(pred_e)) == y_e),
                    "max_cal_proba": mp,
                    "top2_margin": mg,
                    "confidence_bucket": _confidence_bucket(mp) if mp is not None else "unknown",
                    "pnl": float(t["pnl"]),
                    "win": bool(float(t["pnl"]) > 0),
                    "hold_bars": int(t["hold_bars"]),
                    "mae_raw": excursion["mae_raw"],
                    "mfe_raw": excursion["mfe_raw"],
                    "mae_atr": excursion["mae_atr"],
                    "mfe_atr": excursion["mfe_atr"],
                }
            )

        weak_mask = (merged_te["regime_vol_state"].astype(str) == "mid_vol") & (
            merged_te["regime_structure"].astype(str) == "range_bound"
        )
        for j in np.flatnonzero(weak_mask.to_numpy()):
            j = int(j)
            pr = float(pred_te[j])
            weak_bar_rows.append(
                {
                    "slice": slice_label,
                    "window_index": int(wi),
                    "pred_class": int(round(pr)),
                    "label": int(y_te[j]),
                    "max_cal_proba": float(max_p[j]) if np.isfinite(max_p[j]) else None,
                    "top2_margin": float(margin[j]) if np.isfinite(margin[j]) else None,
                }
            )

    return {
        "slice_label": slice_label,
        "labeling_mode": label_mode,
        "n_merged_rows": n,
        "trade_rows": trade_rows,
        "weak_regime_test_bar_rows": weak_bar_rows,
    }


def cohort_pnl_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Per-cohort stats including **total_pnl** (sum of realized trade legs in backtest currency units).
    """
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_hold_bars": None,
        }
    pnls = np.array([float(t["pnl"]) for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pos_hold = np.array([int(t["hold_bars"]) for t in trades], dtype=np.float64)
    sw = float(np.sum(wins)) if wins.size else 0.0
    sl = float(np.sum(losses)) if losses.size else 0.0
    if losses.size == 0:
        pf: float | None = float("inf") if wins.size else None
    else:
        pf = sw / abs(sl)
    return {
        "n_trades": int(len(trades)),
        "total_pnl": float(np.sum(pnls)),
        "win_rate": float(np.mean(pnls > 0)),
        "profit_factor": pf,
        "avg_win": float(np.mean(wins)) if wins.size else None,
        "avg_loss": float(np.mean(losses)) if losses.size else None,
        "avg_hold_bars": float(np.mean(pos_hold)),
    }


def summarize_trade_group(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """PnL stats for a list of trade row dicts (expects ``pnl``, ``win``, ``hold_bars``)."""
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_hold_bars": None,
            "long_n": 0,
            "short_n": 0,
        }
    pnls = np.array([float(t["pnl"]) for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pos_hold = np.array([int(t["hold_bars"]) for t in trades], dtype=np.float64)
    sw = float(np.sum(wins)) if wins.size else 0.0
    sl = float(np.sum(losses)) if losses.size else 0.0
    pf: float | None
    if losses.size == 0:
        pf = float("inf") if wins.size else None
    else:
        pf = sw / abs(sl)
    long_n = sum(1 for t in trades if int(t.get("direction") or 0) > 0)
    short_n = sum(1 for t in trades if int(t.get("direction") or 0) < 0)
    return {
        "n_trades": int(len(trades)),
        "win_rate": float(np.mean(pnls > 0)),
        "profit_factor": pf,
        "avg_win": float(np.mean(wins)) if wins.size else None,
        "avg_loss": float(np.mean(losses)) if losses.size else None,
        "avg_hold_bars": float(np.mean(pos_hold)),
        "long_n": int(long_n),
        "short_n": int(short_n),
    }


def bucket_win_rates(trades: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per confidence bucket: count and win rate."""
    from collections import defaultdict

    acc: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        b = str(t.get("confidence_bucket") or "unknown")
        acc[b].append(bool(t.get("win")))
    out: dict[str, dict[str, float]] = {}
    for b, wins in sorted(acc.items()):
        out[b] = {"n": float(len(wins)), "win_rate": float(np.mean(wins)) if wins else 0.0}
    return out


def weak_bar_signal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate signal quality on mid_vol+range_bound test bars."""
    if not rows:
        return {
            "n_bars": 0,
            "mean_max_cal_proba": None,
            "mean_top2_margin": None,
            "pred_class_counts": {},
            "label_counts": {},
        }
    mp = [r["max_cal_proba"] for r in rows if r.get("max_cal_proba") is not None]
    mg = [r["top2_margin"] for r in rows if r.get("top2_margin") is not None]
    from collections import Counter

    pc = Counter(str(int(r["pred_class"])) for r in rows)
    lc = Counter(str(int(r["label"])) for r in rows)
    return {
        "n_bars": len(rows),
        "mean_max_cal_proba": float(np.mean(mp)) if mp else None,
        "mean_top2_margin": float(np.mean(mg)) if mg else None,
        "pred_class_counts": dict(sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))),
        "label_counts": dict(sorted(lc.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

# public-candidate marker
