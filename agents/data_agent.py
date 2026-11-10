"""
Data agent: inspect OHLCV using existing loaders and schema (no replacement loaders).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.types import DataAgentOutput, DataAgentStepResult
from data.loaders.csv_loader import load_csv
from data.ohlcv_schema import SCHEMA_COLUMNS


def _infer_bar_regularity(timestamp: pd.Series) -> tuple[str, float | None]:
    ts = pd.to_datetime(timestamp, errors="coerce")
    if ts.isna().all() or len(ts) < 3:
        return "unknown", None
    ts = ts.sort_values()
    delta_sec = ts.diff().dt.total_seconds().astype("float64").dropna()
    if delta_sec.empty:
        return "unknown", None
    med = float(delta_sec.median())
    if med <= 0:
        return "time_irregular", med
    cv = float(delta_sec.std() / med) if delta_sec.std() == delta_sec.std() else float("inf")
    if cv < 0.08:
        return "time_regular", med
    return "time_irregular", med


@dataclass
class DataAgentInput:
    """Point to dataset on disk (CSV via existing pipeline loader)."""

    path: str
    source: str = "csv"


def run_data_agent(inp: DataAgentInput) -> DataAgentStepResult:
    """
    Load and summarize the dataset: schema, missing values, time span, bar regularity.

    Raises:
        ValueError: if ``source`` is not supported.
    """
    if inp.source != "csv":
        raise ValueError(f"data_agent supports source='csv' only, got {inp.source!r}")

    df = load_csv(inp.path)
    missing = {c: int(df[c].isna().sum()) for c in SCHEMA_COLUMNS if c in df.columns}
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    dup_ts = 0
    t0 = t1 = None
    med_delta: float | None = None
    bar_kind = "unknown"

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        dup_ts = int(ts.duplicated().sum())
        if ts.notna().any():
            t0 = str(ts.min())
            t1 = str(ts.max())
        bar_kind, med_delta = _infer_bar_regularity(df["timestamp"])

    out = DataAgentOutput(
        schema_columns=tuple(c for c in SCHEMA_COLUMNS if c in df.columns),
        dtypes=dtypes,
        missing_counts=missing,
        n_rows=len(df),
        time_start=t0,
        time_end=t1,
        duplicate_timestamps=dup_ts,
        inferred_bar_type=bar_kind,
        median_bar_delta_seconds=med_delta,
        source_path=str(inp.path),
    )
    return DataAgentStepResult(output=out, ohlcv=df)
