"""
CSV OHLCV loader.

Parses generic comma/tab-separated files into the canonical schema via
`preprocess_ohlcv`. Does not infer bar frequency; use `data.bars` for aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.loaders.base import BaseLoader
from data.preprocessing.clean import preprocess_ohlcv


class CsvOhlcvLoader(BaseLoader):
    """Load OHLCV from a CSV (or delimiter-sniffed text) file."""

    def __init__(self, path: str | Path, **read_csv_kwargs: Any) -> None:
        self._path = Path(path)
        self._read_csv_kwargs = read_csv_kwargs

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> pd.DataFrame:
        kwargs = dict(self._read_csv_kwargs)
        if "sep" not in kwargs and "delimiter" not in kwargs:
            kwargs.setdefault("sep", None)
            kwargs.setdefault("engine", "python")
        df = pd.read_csv(self._path, **kwargs)
        return preprocess_ohlcv(df)


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Convenience one-shot load from CSV."""
    return CsvOhlcvLoader(path, **read_csv_kwargs).load()
