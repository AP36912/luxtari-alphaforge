"""Lean, network-free public experiment using real feature/backtest APIs."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from backtest.config import BacktestConfig, ExecutionAssumptions, ExecutionCosts, PositionSizing
from backtest.engine import BacktestEngine
from backtest.metrics import summarize
from features.config import FeatureConfig
from features.pipeline import build_features
from utils.synthetic_ohlcv import generate_synthetic_ohlcv


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_public_synthetic(*, seed: int = 42, rows: int = 750,
                         output_dir: str | Path = "outputs") -> Path:
    """Run deterministic SMA momentum research and write an auditable JSON report.

    Synthetic results demonstrate reproducibility and chronology safety only.
    They are not evidence of real-market profitability, CME/IBKR equivalence,
    or tradable edge.
    """
    raw = generate_synthetic_ohlcv(rows=rows, seed=seed)
    feature_cfg = FeatureConfig(discovery_enabled=False)
    featured = build_features(raw, feature_cfg, dropna=True, include_ohlcv=True)
    fast = featured["close"].rolling(10, min_periods=10).mean()
    slow = featured["close"].rolling(30, min_periods=30).mean()
    # Signals use only current/past closes; next-bar-open execution prevents same-bar lookahead.
    signal = np.sign((fast - slow).fillna(0.0).to_numpy(dtype=float))
    assumptions = ExecutionAssumptions(
        fill_basis="next_bar_open",
        costs=ExecutionCosts(half_spread=0.0001, slippage=0.0001,
                             commission_rate=0.0002),
        stop_target_fill_policy="stop_level",
    )
    sizing = PositionSizing(mode="fraction_of_equity", fraction_of_equity=0.25)
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        assumptions=assumptions,
        sizing=sizing,
        bars_per_year=24 * 365,
    )
    result = BacktestEngine(cfg).run(featured[["open", "high", "low", "close"]], signal)
    metrics = summarize(result.equity, result.trade_legs, bars_per_year=cfg.bars_per_year)
    input_bytes = raw.to_csv(index=False, float_format="%.12g").encode()
    spec = {
        "seed": seed,
        "rows": rows,
        "strategy": "sma_10_30_sign",
        "instrument": "SYNTH",
        "timeframe": "1h",
        "feature_config": {"discovery_enabled": False, "dropna": True, "include_ohlcv": True},
        "assumptions": assumptions.to_dict(),
        "sizing": {
            "mode": sizing.mode,
            "fixed_shares": sizing.fixed_shares,
            "fraction_of_equity": sizing.fraction_of_equity,
        },
        "initial_capital": cfg.initial_capital,
        "bars_per_year": cfg.bars_per_year,
        "cost_notes": {
            "equity_path_includes_proportional_costs": True,
            "trade_leg_win_rate_profit_factor": "TradeLeg metrics include spread/slippage via fill prices but exclude commissions; equity path includes all configured proportional costs",
            "half_spread": assumptions.costs.half_spread,
            "slippage": assumptions.costs.slippage,
            "commission_rate": assumptions.costs.commission_rate,
        },
        "external_validity": (
            "Synthetic OHLCV only; no claim of profitability, CME equivalence, "
            "IBKR validation, or universal simulation parity."
        ),
        "platform_identity": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
        },
        "metric_definitions": {
            "n_trades": "Count of BacktestEngine TradeLeg closed-lot records with realized pnl.",
            "win_rate_profit_factor": "TradeLeg metrics include spread/slippage via fill prices but exclude commissions; equity path includes all configured proportional costs.",
        },
    }
    payload = {
        **spec,
        "config_digest_sha256": _sha256(json.dumps(spec, sort_keys=True).encode()),
        "input_checksum_sha256": _sha256(input_bytes),
        "feature_rows": len(featured),
        "metrics": dataclasses.asdict(metrics),
        "chronology": "causal rolling features; signal[t] executes at open[t+1]",
        "network_used": False,
        "broker_used": False,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"public_offline_seed_{seed}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
