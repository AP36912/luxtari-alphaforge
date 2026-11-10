"""
Meta-labels: secondary targets derived from primary triple-barrier outcomes.

Meta labels are defined **only** from the primary label column (and optionally
future extensions); they do not peek at raw prices beyond what the primary
already encoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeling.config import MetaLabelConfig


def apply_meta_label(primary: pd.Series, cfg: MetaLabelConfig) -> pd.Series:
    """
    Build a meta-label series aligned to ``primary``.

    NaNs in ``primary`` propagate as NaN in the meta label.
    """
    out = pd.Series(np.nan, index=primary.index, dtype=np.float64)
    m = primary.notna()
    if not m.any():
        return out

    p = primary.loc[m]
    if cfg.mode == "profit_first":
        out.loc[m] = (p == 1).astype(np.float64)
    elif cfg.mode == "decisive":
        out.loc[m] = (p.abs() == 1).astype(np.float64)
    elif cfg.mode == "avoid_loss":
        out.loc[m] = (p != -1).astype(np.float64)
    else:
        raise ValueError(f"Unknown meta mode: {cfg.mode!r}")
    return out
