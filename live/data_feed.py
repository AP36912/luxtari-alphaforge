"""
Polling-style market data ingestion with canonical OHLCV schema.

Adapters (CSV poll, broker WebSocket, etc.) can subclass :class:`AbstractDataFeed`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data.ohlcv_schema import SCHEMA_COLUMNS, enforce_ohlcv_schema
from live.bar_validation import validate_live_bar_row


@dataclass
class BarMergeReport:
    """Diagnostics after merging new bars into a buffer."""

    n_incoming: int = 0
    n_duplicates_dropped: int = 0
    n_out_of_order_corrected: int = 0
    n_gaps_detected: int = 0
    last_timestamp: pd.Timestamp | None = None
    n_quarantined: int = 0
    notes: list[str] = field(default_factory=list)


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return sorted canonical OHLCV (duplicates preserved until merge)."""
    return enforce_ohlcv_schema(df, sort=True)


def merge_ohlcv_buffer(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    *,
    expected_freq: pd.Timedelta | None = None,
    quarantine: Any = None,
) -> tuple[pd.DataFrame, BarMergeReport]:
    """
    Append ``incoming`` to ``existing``, drop duplicate timestamps (keep last),
    sort stably by time.

    Detects missing bars when ``expected_freq`` is set (simple consecutive diff check).
    """
    rep = BarMergeReport()
    if incoming is None or len(incoming) == 0:
        out = existing.copy() if existing is not None else pd.DataFrame(columns=list(SCHEMA_COLUMNS))
        return out, rep

    if len(incoming) > 1:
        ts_raw = pd.to_datetime(incoming["timestamp"], errors="coerce")
        if not ts_raw.is_monotonic_increasing:
            rep.n_out_of_order_corrected = 1
            rep.notes.append("incoming_batch_had_out_of_order_timestamps")

    keep_idx: list[int] = []
    for i in range(len(incoming)):
        vr = validate_live_bar_row(incoming.iloc[i])
        if vr.ok:
            keep_idx.append(i)
        else:
            rep.n_quarantined += 1
            if quarantine is not None:
                quarantine.write_invalid(
                    incoming.iloc[i].to_dict(),
                    vr,
                    context="merge_ohlcv_buffer",
                )
    if not keep_idx:
        out = existing.copy() if existing is not None else pd.DataFrame(columns=list(SCHEMA_COLUMNS))
        return out, rep

    inc = normalize_ohlcv_frame(incoming.iloc[keep_idx].reset_index(drop=True))
    rep.n_incoming = len(inc)

    if existing is None or len(existing) == 0:
        base = pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    else:
        base = normalize_ohlcv_frame(existing)

    if len(base) == 0:
        combined = inc.copy()
    else:
        combined = pd.concat([base, inc], ignore_index=True)
    before = len(combined)
    combined = combined.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    rep.n_duplicates_dropped = before - len(combined)
    if len(combined):
        rep.last_timestamp = pd.Timestamp(combined["timestamp"].iloc[-1])

    if expected_freq is not None and len(combined) > 1:
        diffs = combined["timestamp"].diff().dropna()
        # allow small tolerance
        tol = expected_freq * 0.01
        gaps = diffs > (expected_freq + tol)
        rep.n_gaps_detected = int(gaps.sum())
        if rep.n_gaps_detected:
            rep.notes.append(f"detected_{rep.n_gaps_detected}_gaps_vs_expected_freq")

    return combined, rep


class AbstractDataFeed(ABC):
    """Extension point for broker / file / API polling implementations."""

    @abstractmethod
    def fetch_new_bars(self) -> pd.DataFrame:
        """Return zero or more new rows with ``SCHEMA_COLUMNS``."""

    def poll(self) -> pd.DataFrame:
        """Default: same as :meth:`fetch_new_bars` (override if needed)."""
        return self.fetch_new_bars()


class InMemoryPollingFeed(AbstractDataFeed):
    """
    Test / simulation feed: caller pushes batches via :meth:`push`, :meth:`poll` drains queue.
    """

    def __init__(self) -> None:
        self._queue: list[pd.DataFrame] = []

    def push(self, batch: pd.DataFrame) -> None:
        self._queue.append(batch)

    def fetch_new_bars(self) -> pd.DataFrame:
        if not self._queue:
            return pd.DataFrame(columns=list(SCHEMA_COLUMNS))
        parts = self._queue
        self._queue = []
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


class StaticSnapshotFeed(AbstractDataFeed):
    """Returns the same frame once (or every poll) — useful for deterministic tests."""

    def __init__(self, df: pd.DataFrame, *, repeat: bool = False) -> None:
        self._df = normalize_ohlcv_frame(df)
        self._repeat = repeat
        self._consumed = False

    def fetch_new_bars(self) -> pd.DataFrame:
        if self._consumed and not self._repeat:
            return pd.DataFrame(columns=list(SCHEMA_COLUMNS))
        self._consumed = True
        return self._df.copy()
