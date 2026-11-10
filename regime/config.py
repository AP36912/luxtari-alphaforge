"""
Configuration for market regime detection, sessions (e.g. XAUUSD), and trade filters.

All session windows are interpreted in ``session_timezone`` using wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

DetectionMethod = Literal["rules", "cluster_offline"]


@dataclass
class SessionWindow:
    """Half-open wall-clock interval ``[start, end)`` in ``session_timezone``."""

    name: str
    start_hour: int
    start_minute: int = 0
    end_hour: int = 0
    end_minute: int = 0

    def __post_init__(self) -> None:
        for h in (self.start_hour, self.end_hour):
            if not 0 <= h <= 23:
                raise ValueError(f"hour out of range: {h}")
        for m in (self.start_minute, self.end_minute):
            if not 0 <= m <= 59:
                raise ValueError(f"minute out of range: {m}")


@dataclass
class SessionCalendarConfig:
    """
    FX-style session tagging (defaults tuned for liquid XAU / FX overlap logic in UTC).

    Override windows for your broker or data timezone. ``rollover`` is checked first,
    then ``london_ny_overlap``, then ``london``, ``new_york``, ``asia``, else ``off_hours``.
    """

    session_timezone: str = "UTC"
    """IANA timezone name (e.g. ``UTC``, ``America/New_York``)."""
    assume_timestamp_utc_if_naive: bool = True
    """If True, naive ``timestamp`` is treated as UTC before converting to ``session_timezone``."""

    rollover: SessionWindow = field(
        default_factory=lambda: SessionWindow("rollover", 21, 45, 22, 30)
    )
    london_ny_overlap: SessionWindow = field(
        default_factory=lambda: SessionWindow("london_ny_overlap", 13, 30, 16, 30)
    )
    london: SessionWindow = field(
        default_factory=lambda: SessionWindow("london", 8, 0, 16, 30)
    )
    new_york: SessionWindow = field(
        default_factory=lambda: SessionWindow("new_york", 13, 30, 21, 0)
    )
    asia: SessionWindow = field(
        default_factory=lambda: SessionWindow("asia", 0, 0, 8, 0)
    )


@dataclass
class RegimeDetectionParams:
    """Thresholds for rule-based structure / volatility labels (causal rolling features)."""

    vol_short_window: int = 10
    vol_long_window: int = 40
    vol_quantile_window: int = 120
    vol_high_quantile: float = 0.75
    vol_low_quantile: float = 0.25

    atr_short_period: int = 7
    atr_long_period: int = 21
    atr_expand_ratio: float = 1.25
    """``ATR_short / ATR_long`` above this → ``breakout_expand`` candidate."""

    range_window: int = 20
    compression_range_atr_ratio: float = 0.85
    """(High-low range over window) / ATR below this × scale → ``compressed``."""

    slope_window: int = 30
    slope_trending_atr_ratio: float = 0.12
    """|OLS slope of close| / ATR above this → trending candidate."""
    slope_range_atr_ratio: float = 0.04
    """Below this (and not compressed/expand) → ``range_bound``."""

    momentum_window: int = 10
    momentum_confirm_ratio: float = 0.15
    """|pct_change(momentum_window)| must exceed this for trending confirmation."""

    cluster_n_clusters: int = 4
    cluster_random_state: int = 0
    cluster_min_train_rows: int = 200


@dataclass
class RegimeFilterConfig:
    """Gate trading by session / volatility state (used when ``apply_to_signals`` is true)."""

    enabled: bool = False
    apply_to_signals: bool = False
    """If True, zero signals where filters fail (holdout backtest)."""
    allowed_sessions: tuple[str, ...] = ()
    """Empty = all sessions allowed. Names: ``asia``, ``london``, ``new_york``, ``london_ny_overlap``, etc."""
    allowed_volatility_states: tuple[str, ...] = ()
    """Empty = all. Values: ``low_vol``, ``mid_vol``, ``high_vol``."""
    require_both: bool = True
    """If True, bar must pass session AND vol filters when both lists non-empty."""
    block_high_vol_entries: bool = False
    """If True (walk-forward / holdout): block only **new entries** when ``regime_vol_state == high_vol``."""
    block_trending_down_entries: bool = False
    """If True: block only **new entries** when ``regime_structure == trending_down`` (research structure tag)."""
    block_lowvol_rangebound_london: bool = False
    """If True: block new entries when vol==low_vol AND structure==range_bound AND session==london."""
    exit_lowvol_neutral_long_defensive_enabled: bool = False
    """If True: apply defensive signal->flat exit only for low_vol|neutral|long entries."""
    exit_lowvol_neutral_long_atr_period: int = 14
    """ATR period for defensive low-vol exit threshold."""
    exit_lowvol_neutral_long_atr_multiple: float = 0.75
    """Exit threshold: entry_price - atr_multiple * ATR(period)."""


@dataclass
class RegimeConfig:
    """Top-level regime + session settings for experiments."""

    enabled: bool = False
    add_feature_columns: bool = True
    """Append numeric regime id columns to the modeling feature matrix."""
    detection_method: DetectionMethod = "rules"
    include_segment_metrics: bool = True
    """Add holdout backtest metrics sliced by session / regime (see ``backtest.segmentation``)."""
    sessions: SessionCalendarConfig = field(default_factory=SessionCalendarConfig)
    detection: RegimeDetectionParams = field(default_factory=RegimeDetectionParams)
    filters: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)


def regime_config_from_dict(raw: dict[str, Any] | None) -> RegimeConfig:
    """Build :class:`RegimeConfig` from YAML fragment."""
    if not raw:
        return RegimeConfig()
    allowed_top = {"enabled", "sessions", "detection", "filters"}
    # RegimeConfig may have more fields; include dataclass names when present.
    allowed_top |= {f.name for f in fields(RegimeConfig)}
    unknown_top = sorted(str(k) for k in raw if k not in allowed_top)
    if unknown_top:
        raise ValueError("unknown regime key(s): " + ", ".join(unknown_top))
    sess_raw = raw.get("sessions") or {}
    det_raw = raw.get("detection") or {}
    filt_raw = raw.get("filters") or {}
    if not isinstance(sess_raw, dict) or not isinstance(det_raw, dict) or not isinstance(filt_raw, dict):
        raise ValueError("regime sessions/detection/filters must be mappings")

    def _sw(prefix: str, d: dict[str, Any], default: SessionWindow) -> SessionWindow:
        key = prefix
        block = d.get(key)
        if not isinstance(block, dict):
            return default
        return SessionWindow(
            name=str(block.get("name", default.name)),
            start_hour=int(block.get("start_hour", default.start_hour)),
            start_minute=int(block.get("start_minute", default.start_minute)),
            end_hour=int(block.get("end_hour", default.end_hour)),
            end_minute=int(block.get("end_minute", default.end_minute)),
        )

    base_cal = SessionCalendarConfig()
    cal = SessionCalendarConfig(
        session_timezone=str(sess_raw.get("session_timezone", base_cal.session_timezone)),
        assume_timestamp_utc_if_naive=bool(
            sess_raw.get("assume_timestamp_utc_if_naive", base_cal.assume_timestamp_utc_if_naive)
        ),
        rollover=_sw("rollover", sess_raw, base_cal.rollover),
        london_ny_overlap=_sw("london_ny_overlap", sess_raw, base_cal.london_ny_overlap),
        london=_sw("london", sess_raw, base_cal.london),
        new_york=_sw("new_york", sess_raw, base_cal.new_york),
        asia=_sw("asia", sess_raw, base_cal.asia),
    )
    # Allow flat keys rollover_start_hour in sess_raw - optional shortcut omitted for brevity

    allowed_sess_keys = {
        "session_timezone",
        "assume_timestamp_utc_if_naive",
        "rollover",
        "london_ny_overlap",
        "london",
        "new_york",
        "asia",
    }
    unknown_sess = sorted(str(k) for k in sess_raw if k not in allowed_sess_keys)
    if unknown_sess:
        raise ValueError("unknown regime.sessions key(s): " + ", ".join(unknown_sess))

    det_fields = {f.name for f in fields(RegimeDetectionParams)}
    unknown_det = sorted(str(k) for k in det_raw if k not in det_fields)
    if unknown_det:
        raise ValueError("unknown regime.detection key(s): " + ", ".join(unknown_det))
    det_base = {f.name: getattr(RegimeDetectionParams(), f.name) for f in fields(RegimeDetectionParams)}
    det_kw = {k: v for k, v in det_raw.items() if k in det_fields}
    detection = RegimeDetectionParams(**{**det_base, **det_kw})

    filt_fields = {f.name for f in fields(RegimeFilterConfig)}
    unknown_filt = sorted(str(k) for k in filt_raw if k not in filt_fields)
    if unknown_filt:
        raise ValueError("unknown regime.filters key(s): " + ", ".join(unknown_filt))
    filt_base = {f.name: getattr(RegimeFilterConfig(), f.name) for f in fields(RegimeFilterConfig)}
    allowed_sess = filt_raw.get("allowed_sessions")
    allowed_vol = filt_raw.get("allowed_volatility_states")
    filt_kw = {k: v for k, v in filt_raw.items() if k in filt_fields and k not in ("allowed_sessions", "allowed_volatility_states")}
    if allowed_sess is not None:
        filt_kw["allowed_sessions"] = tuple(str(x) for x in allowed_sess)
    if allowed_vol is not None:
        filt_kw["allowed_volatility_states"] = tuple(str(x) for x in allowed_vol)
    filters = RegimeFilterConfig(**{**filt_base, **filt_kw})

    top_fields = {f.name for f in fields(RegimeConfig)}
    top_kw = {k: v for k, v in raw.items() if k in top_fields and k not in ("sessions", "detection", "filters")}
    return RegimeConfig(
        sessions=cal,
        detection=detection,
        filters=filters,
        **top_kw,
    )
