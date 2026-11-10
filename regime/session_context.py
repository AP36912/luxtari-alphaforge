"""
Timezone-aware FX / metals session tagging (Asia, London, New York, overlap, rollover).

Deterministic: same timestamps always yield the same session label for a fixed :class:`SessionCalendarConfig`.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from regime.config import SessionCalendarConfig, SessionWindow


def _minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def _in_half_open_window(t_minutes: int, win: SessionWindow) -> bool:
    """``[start, end)`` in minutes from midnight; supports wrap past midnight."""
    a = win.start_hour * 60 + win.start_minute
    b = win.end_hour * 60 + win.end_minute
    if a <= b:
        return a <= t_minutes < b
    return t_minutes >= a or t_minutes < b


def session_tag_for_row(wall_time: time, cfg: SessionCalendarConfig) -> str:
    """Single-row session label using priority: rollover → overlap → london → new_york → asia → off_hours."""
    m = _minutes_since_midnight(wall_time)
    if _in_half_open_window(m, cfg.rollover):
        return "rollover"
    if _in_half_open_window(m, cfg.london_ny_overlap):
        return "london_ny_overlap"
    if _in_half_open_window(m, cfg.london):
        return "london"
    if _in_half_open_window(m, cfg.new_york):
        return "new_york"
    if _in_half_open_window(m, cfg.asia):
        return "asia"
    return "off_hours"


def compute_session_tags(df: pd.DataFrame, cfg: SessionCalendarConfig) -> pd.Series:
    """
    Return a string ``session`` column aligned with ``df`` (same index).

    Requires ``timestamp`` column. NaT timestamps → ``off_hours``.
    Naive timestamps are interpreted as UTC when ``assume_timestamp_utc_if_naive`` is True.
    """
    if "timestamp" not in df.columns:
        raise KeyError("compute_session_tags requires column 'timestamp'")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.dt.tz is None:
        if not cfg.assume_timestamp_utc_if_naive:
            raise ValueError(
                "Timestamps are timezone-naive; set SessionCalendarConfig.assume_timestamp_utc_if_naive=True "
                "or provide tz-aware timestamps."
            )
        ts = ts.dt.tz_localize("UTC")
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(cfg.session_timezone)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid session_timezone {cfg.session_timezone!r}") from exc
    wall = ts.dt.tz_convert(tz)
    out: list[str] = []
    for w in wall:
        if pd.isna(w):
            out.append("off_hours")
            continue
        t = w.to_pydatetime().time() if hasattr(w, "to_pydatetime") else pd.Timestamp(w).time()
        out.append(session_tag_for_row(t, cfg))
    return pd.Series(out, index=df.index, dtype="object")


def session_tag_to_id(tags: pd.Series, *, order: tuple[str, ...] | None = None) -> np.ndarray:
    """Stable integer encoding for ML features (unknown tags → -1)."""
    default_order = (
        "asia",
        "london",
        "new_york",
        "london_ny_overlap",
        "rollover",
        "off_hours",
    )
    order = order or default_order
    mapping = {name: i for i, name in enumerate(order)}
    arr = np.full(len(tags), -1, dtype=np.int64)
    for i, v in enumerate(tags.astype(str).values):
        arr[i] = int(mapping.get(str(v), -1))
    return arr
