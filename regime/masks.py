"""
Named bar masks on holdout (or full) frames with regime/session columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import RegimeFilterConfig
from regime.filters import trading_allowed_mask


def build_holdout_segment_masks(
    df: pd.DataFrame,
    filt: RegimeFilterConfig,
) -> dict[str, np.ndarray]:
    """
    Standard comparison groups for reporting.

    Includes ``all_bars``, session slices, volatility slices, conjunctions, and
    ``configured_gates`` from :class:`RegimeFilterConfig` (empty allowed lists → no extra restriction).
    """
    n = len(df)
    if "regime_session" not in df.columns or "regime_vol_state" not in df.columns:
        return {"all_bars": np.ones(n, dtype=bool)}

    sess = df["regime_session"]
    vol = df["regime_vol_state"]
    masks: dict[str, np.ndarray] = {}
    masks["all_bars"] = np.ones(n, dtype=bool)
    masks["session_exclude_rollover"] = (~sess.astype(str).eq("rollover")).to_numpy()
    masks["session_asia"] = sess.astype(str).eq("asia").to_numpy()
    masks["session_london"] = sess.astype(str).eq("london").to_numpy()
    masks["session_new_york"] = sess.astype(str).eq("new_york").to_numpy()
    masks["session_london_ny_overlap"] = sess.astype(str).eq("london_ny_overlap").to_numpy()
    liquid = frozenset({"asia", "london", "new_york", "london_ny_overlap"})
    masks["session_liquid_major"] = sess.astype(str).isin(liquid).to_numpy()

    masks["vol_high"] = vol.astype(str).eq("high_vol").to_numpy()
    masks["vol_low"] = vol.astype(str).eq("low_vol").to_numpy()
    masks["vol_mid"] = vol.astype(str).eq("mid_vol").to_numpy()

    masks["session_overlap_and_high_vol"] = masks["session_london_ny_overlap"] & masks["vol_high"]

    sess_tuple = filt.allowed_sessions if filt.allowed_sessions else tuple()
    vol_tuple = filt.allowed_volatility_states if filt.allowed_volatility_states else tuple()
    masks["configured_gates"] = trading_allowed_mask(
        sess,
        vol,
        allowed_sessions=sess_tuple if sess_tuple else None,
        allowed_volatility_states=vol_tuple if vol_tuple else None,
        require_both=filt.require_both,
    )
    return masks
