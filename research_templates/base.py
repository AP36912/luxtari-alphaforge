"""Shared experiment skeleton for public research templates."""
from __future__ import annotations

from typing import Any

PUBLIC_RESEARCH_BASE: dict[str, Any] = {
    "metadata": {
        "asset": "SYNTH",
        "timeframe": "M5",
        "notes": "Public research template base — set data.path to a synthetic or licensed OHLCV CSV.",
    },
    "features": {
        "discovery_enabled": False,
        "price_enabled": True,
        "technical_enabled": True,
        "statistical_enabled": True,
        "microstructure_enabled": True,
    },
    "labeling": {"mode": "triple_barrier"},
    "model": {"name": "random_forest"},
    "execution": {
        "execution_mode": "paper",
    },
    "backtest": {
        "initial_capital": 100000.0,
        "bars_per_year": 252,
        "costs": {"half_spread": 0.0, "slippage": 0.0, "commission_rate": 0.0},
        "sizing": {"mode": "fixed_shares", "fixed_shares": 1.0},
    },
}

# Compatibility aliases used by merge helpers / older call sites
SYNTH_RESEARCH_BASE = PUBLIC_RESEARCH_BASE

import copy
from typing import Any as _Any


def deep_merge(base: dict[str, _Any], overlay: dict[str, _Any]) -> dict[str, _Any]:
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out
