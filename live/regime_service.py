"""
Real-time regime / session evaluation using research :mod:`regime` helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.filters import trading_allowed_mask

from live.live_types import RegimeSnapshot


class RegimeService:
    """
    Interprets the latest row of a frame already processed by
    :func:`regime.integration.attach_regime_columns`.
    """

    def __init__(self, cfg: RegimeConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> RegimeConfig:
        return self._cfg

    def snapshot_from_row(self, row: pd.Series) -> RegimeSnapshot:
        if not self._cfg.enabled:
            return RegimeSnapshot(
                session="any",
                vol_state="any",
                structure="any",
                trading_allowed=True,
            )
        sess = str(row.get("regime_session", "unknown"))
        vol = str(row.get("regime_vol_state", "unknown"))
        struct = str(row.get("regime_structure", "unknown"))
        allowed = bool(
            trading_allowed_mask(
                np.array([sess]),
                np.array([vol]),
                allowed_sessions=self._cfg.filters.allowed_sessions or None,
                allowed_volatility_states=self._cfg.filters.allowed_volatility_states or None,
                require_both=self._cfg.filters.require_both,
            )[0]
        )
        # If filters disabled in config, mask treats empty tuples as allow-all
        if not self._cfg.filters.enabled:
            allowed = True
        return RegimeSnapshot(session=sess, vol_state=vol, structure=struct, trading_allowed=allowed)
