"""
Abstract loader contracts.

Loaders return standardized OHLCV frames; lookahead must not be introduced at
ingest (e.g. using revised fundamentals without as-of dates).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """Base class for file- or API-backed OHLCV loaders."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame with columns ``timestamp, open, high, low, close, volume``."""
        raise NotImplementedError

    @property
    def path(self) -> Path | None:
        """Optional filesystem path for manifesting (subclasses may override)."""
        return None
