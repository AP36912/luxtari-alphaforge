"""
Structured experiment configuration (from YAML / dict).

Paths in ``data.path`` are relative to the **project root** unless absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from backtest.config import (
    BacktestConfig,
    ExecutionAssumptions,
    ExecutionCosts,
    PositionSizing,
)
from features.config import FeatureConfig
from labeling.config import TripleBarrierConfig
from models.config import ModelConfig
from regime.config import RegimeConfig, regime_config_from_dict


@dataclass
class DataConfig:
    """Input OHLCV source."""

    source: str = "csv"
    path: str = ""


@dataclass
class ValidationConfig:
    """Purged CV + holdout fraction for OOS backtest."""

    n_splits: int = 3
    label_horizon: int = 5
    embargo: int = 0
    train_fraction: float = 0.72


@dataclass
class WalkForwardCalibrationConfig:
    """
    Defaults for :func:`scripts.run_walk_forward` probability calibration.

    Fit slice + calibration tail are time-ordered inside each train window only;
    test rows are never used. CLI flags override these when explicitly passed.
    """

    method: Literal["none", "sigmoid", "isotonic"] = "none"
    holdout_fraction: float = 0.15
    min_cal_rows: int = 50
    min_fit_rows: int = 50


@dataclass
class OutputConfig:
    """Artifact layout under ``outputs/runs/<run_id>/``."""

    run_id: str | None = None


@dataclass
class AuditConfig:
    """Pipeline audit gates (warnings vs strict errors)."""

    strict: bool = False
    """If True, any audit *error* finding aborts the run."""
    max_abs_corr_vs_next_return: float = 0.995
    """Leakage screen: flag |corr(feature, next return)| at or above this level."""
    check_timestamps: bool = True
    check_leakage: bool = True
    check_splits: bool = True
    check_backtest_alignment: bool = True
    check_numpy_seed_sanity: bool = True


@dataclass
class TrackingConfig:
    """When ``enabled``, each ``run_experiment`` appends a row to ``db_path`` (SQLite)."""

    enabled: bool = False
    db_path: str = "outputs/experiments.db"
    """Path relative to project root unless absolute."""
    search_batch_id: str | None = None
    """Optional grouping id for hyperparameter search trials."""
    search_trial_index: int | None = None
    """Trial index within ``search_batch_id`` (0-based)."""


@dataclass
class LiveOpsConfig:
    """Operational knobs for live paper layer (Phase 16.5)."""

    quarantine_enabled: bool = True
    quarantine_dir: str = "outputs/live/quarantine"
    soak_checkpoint_every_steps: int = 100
    soak_max_steps_default: int = 1000


@dataclass
class ExecutionConfig:
    """Public offline execution config. Only ``paper``, ``dry_run``, and ``sim`` are supported.

    Real brokers, including IBKR, are future work and are intentionally absent.
    ``kill_switch_enabled`` enables monitoring; only ``emergency_stop`` or a present
    kill-switch file halts execution.
    """

    execution_mode: str = "paper"
    use_execution_gateway: bool = False
    initial_cash: float = 100_000.0
    contract_size: float = 1.0
    emergency_stop: bool = False
    kill_switch_enabled: bool = False
    kill_switch_file_path: str | None = None
    symbol_allowlist: list[str] = field(default_factory=list)
    symbol_denylist: list[str] = field(default_factory=list)
    stale_signal_max_age_seconds: float | None = 300.0
    max_daily_loss_fraction: float | None = None
    max_position_size: float | None = None
    max_concurrent_positions: int | None = None
    max_notional: float | None = None
    duplicate_order_cooldown_seconds: float = 0.0
    order_idempotency_ttl_seconds: float = 60.0
    manual_approval_required: bool = False
    manual_approval_bypass_paper: bool = True
    audit_jsonl_path: str = "outputs/execution_audit/events.jsonl"
    audit_markdown_dir: str = "outputs/reports/execution"
    approval_queue_dir: str = "outputs/execution_audit/pending"
    signal_min_confidence_for_entry: float | None = None
    signal_min_confidence_for_entry_long: float | None = None
    signal_min_confidence_for_entry_short: float | None = None
    signal_flat_if_max_proba_below: float | None = None
    signal_allow_neutral_class: bool = False
    signal_neutral_lean_threshold: float = 0.0
    signal_class_mapping_override: dict[int, int] | None = None


def _parse_class_mapping_override(raw_val: Any) -> dict[int, int] | None:
    if raw_val is None:
        return None
    if isinstance(raw_val, dict) and raw_val:
        out: dict[int, int] = {}
        for k, v in raw_val.items():
            out[int(k)] = int(v)
        return out
    return None


def _optional_int_field(raw: dict[str, Any], key: str, base: int | None) -> int | None:
    if key not in raw:
        return base
    v = raw[key]
    if v is None or v == "":
        return None
    return int(v)


def _coerce_signal_min_confidence_for_entry(v: Any) -> float | None:
    """``<= 0`` or unset means no entry-confidence gate (validator requires (0, 1] when active)."""
    if v is None:
        return None
    x = float(v)
    if x <= 0.0:
        return None
    return x


def execution_config_from_dict(raw: dict[str, Any] | None) -> ExecutionConfig:
    """Parse the public execution surface and reject removed live-broker settings."""
    if not raw:
        return ExecutionConfig()
    if not isinstance(raw, dict):
        raise ValueError("execution must be a mapping")

    forbidden_exact = {
        "broker_live_enabled", "broker_submit_enabled", "dry_run_shadow_mode",
        "broker_readonly_poll_seconds", "max_broker_submit_retries",
        "manual_approval_bypass_readonly",
    }
    forbidden_prefixes = ("removed_", "mt5", "demo_lifecycle", "automation_")
    forbidden = sorted(
        str(k) for k in raw
        if str(k) in forbidden_exact or str(k).lower().startswith(forbidden_prefixes)
    )
    if forbidden:
        raise ValueError(
            "forbidden public execution key(s): " + ", ".join(forbidden)
            + "; live/MT5/demo automation is not supported"
        )

    from broker.mode import normalize_execution_mode

    values = dict(raw)
    values["execution_mode"] = normalize_execution_mode(
        str(values.get("execution_mode", "paper"))
    ).value
    allowed = {f.name for f in fields(ExecutionConfig)}
    unknown = sorted(str(k) for k in values if k not in allowed)
    if unknown:
        raise ValueError("unknown public execution key(s): " + ", ".join(unknown))

    for key in ("symbol_allowlist", "symbol_denylist"):
        if key in values and not isinstance(values[key], list):
            raise ValueError(f"execution.{key} must be a list")
    optional_float = {
        "stale_signal_max_age_seconds", "max_daily_loss_fraction",
        "max_position_size", "max_notional", "signal_min_confidence_for_entry",
        "signal_min_confidence_for_entry_long", "signal_min_confidence_for_entry_short",
        "signal_flat_if_max_proba_below",
    }
    for key in optional_float:
        if key in values and values[key] not in (None, ""):
            values[key] = float(values[key])
    if "max_concurrent_positions" in values and values["max_concurrent_positions"] is not None:
        values["max_concurrent_positions"] = int(values["max_concurrent_positions"])
    if "signal_class_mapping_override" in values:
        values["signal_class_mapping_override"] = _parse_class_mapping_override(
            values["signal_class_mapping_override"]
        )
    return ExecutionConfig(**values)

def live_ops_config_from_dict(raw: dict[str, Any] | None) -> LiveOpsConfig:
    if not raw:
        return LiveOpsConfig()
    kw = _pick_dataclass_kwargs(LiveOpsConfig, raw, path="live_ops")
    return LiveOpsConfig(**kw)


def tracking_config_from_dict(raw: dict | None) -> TrackingConfig:
    if not raw:
        return TrackingConfig()
    kw = _pick_dataclass_kwargs(TrackingConfig, raw, path="tracking")
    return TrackingConfig(**kw)


@dataclass
class ExperimentConfig:
    """Full experiment definition."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: dict[str, Any] = field(default_factory=dict)
    labeling: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="random_forest"))
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    walk_forward_calibration: WalkForwardCalibrationConfig = field(
        default_factory=WalkForwardCalibrationConfig
    )
    backtest: dict[str, Any] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form run metadata (asset, timeframe, notes) for reports and hashing."""
    audit: AuditConfig = field(default_factory=AuditConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    live_ops: LiveOpsConfig = field(default_factory=LiveOpsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    source_path: Path | None = field(default=None, repr=False, compare=False)


def _pick_dataclass_kwargs(cls: type, d: dict[str, Any], *, path: str = "config") -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    unknown = sorted(str(k) for k in d if k not in names)
    if unknown:
        raise ValueError(f"unknown {path} key(s): " + ", ".join(unknown))
    return {k: v for k, v in d.items() if k in names}


def feature_config_from_overrides(overrides: dict[str, Any] | None) -> FeatureConfig:
    base = FeatureConfig()
    if not overrides:
        return base
    kw = _pick_dataclass_kwargs(FeatureConfig, overrides, path="features")
    return replace(base, **kw)


def triple_barrier_config_from_dict(d: dict[str, Any]) -> TripleBarrierConfig:
    inner = d.get("triple_barrier") or d
    base_fields = {f.name for f in fields(TripleBarrierConfig)}
    kwargs = {k: v for k, v in inner.items() if k in base_fields}
    return TripleBarrierConfig(**kwargs)


def model_config_from_dict(d: dict[str, Any]) -> ModelConfig:
    allowed = {"name", "extra_params"}
    unknown = sorted(str(k) for k in d if k not in allowed)
    if unknown:
        raise ValueError("unknown model key(s): " + ", ".join(unknown))
    name = d.get("name", "random_forest")
    extra = dict(d.get("extra_params") or {})
    return ModelConfig(name=str(name), extra_params=extra)


def execution_assumptions_from_backtest_dict(d: dict[str, Any]) -> ExecutionAssumptions:
    """Build :class:`ExecutionAssumptions` from a ``backtest:`` YAML fragment (legacy + nested)."""
    ex = d.get("execution")
    if isinstance(ex, dict):
        merged = dict(ex)
        if "costs" not in merged and d.get("costs"):
            merged["costs"] = d["costs"]
        return ExecutionAssumptions.from_dict(merged)
    raw_out: dict[str, Any] = {}
    for k in ("fill_basis", "stop_target_fill_policy"):
        if d.get(k) is not None:
            raw_out[k] = d[k]
    if d.get("costs"):
        raw_out["costs"] = d["costs"]
    return ExecutionAssumptions.from_dict(raw_out or None)


def backtest_config_from_dict(d: dict[str, Any]) -> BacktestConfig:
    initial = float(d.get("initial_capital", 100_000.0))
    bars_py = int(d.get("bars_per_year", 252))
    sizing_d = d.get("sizing") or {}
    mode = sizing_d.get("mode", "fixed_shares")
    sizing = PositionSizing(
        mode=mode,
        fixed_shares=float(sizing_d.get("fixed_shares", 1.0)),
        fraction_of_equity=float(sizing_d.get("fraction_of_equity", 0.99)),
    )
    assumptions = execution_assumptions_from_backtest_dict(d)
    return BacktestConfig(
        initial_capital=initial,
        assumptions=assumptions,
        sizing=sizing,
        bars_per_year=bars_py,
        stop_loss_pct=(None if d.get("stop_loss_pct") is None else float(d["stop_loss_pct"])),
        take_profit_pct=(None if d.get("take_profit_pct") is None else float(d["take_profit_pct"])),
    )


def walk_forward_calibration_from_dict(raw: dict[str, Any] | None) -> WalkForwardCalibrationConfig:
    """Parse ``walk_forward_calibration:`` YAML fragment."""
    if not raw:
        return WalkForwardCalibrationConfig()
    kw = _pick_dataclass_kwargs(WalkForwardCalibrationConfig, raw, path="walk_forward_calibration")
    method = kw.get("method", WalkForwardCalibrationConfig().method)
    if method not in ("none", "sigmoid", "isotonic"):
        raise ValueError(
            f"walk_forward_calibration.method must be none|sigmoid|isotonic, got {method!r}"
        )
    return WalkForwardCalibrationConfig(**kw)


_FORBIDDEN_PUBLIC_KEY_PREFIXES = (
    "mt5",
    "bridge",
    "broker_live",
    "broker_submit",
    "broker_credentials",
    "automation_",
    "demo_lifecycle_",
    "removed_",
)
_FORBIDDEN_PUBLIC_KEY_EXACT = {
    "credentials",
    "credential",
    "password",
    "api_key",
    "api_token",
    "access_key",
    "access_token",
    "client_secret",
    "private_key",
    "secret",
    "token",
}

def _reject_forbidden_keys(value: Any, *, path: str) -> None:
    """Reject removed broker/credential keys recursively in execution-related sections."""
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_path = f"{path}.{raw_key}"
            if key in _FORBIDDEN_PUBLIC_KEY_EXACT or key.startswith(_FORBIDDEN_PUBLIC_KEY_PREFIXES):
                raise ValueError(f"forbidden public configuration key (legacy live/broker settings): {child_path}")
            _reject_forbidden_keys(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def experiment_config_from_dict(raw: dict[str, Any], *, source_path: Path | None = None) -> ExperimentConfig:
    """Build :class:`ExperimentConfig` from a YAML-loaded dict, failing closed on schema drift."""
    if not isinstance(raw, dict):
        raise ValueError("experiment configuration must be a mapping")
    allowed_top = {f.name for f in fields(ExperimentConfig)} - {"source_path"}
    allowed_top.add("live")  # legacy alias for offline live_ops only
    unknown = sorted(str(key) for key in raw if key not in allowed_top)
    if unknown:
        raise ValueError("unknown top-level configuration key(s): " + ", ".join(unknown))
    # Reject credential/broker residue anywhere in the raw tree (including metadata/features).
    _reject_forbidden_keys(raw, path="config")
    data = DataConfig(**_pick_dataclass_kwargs(DataConfig, raw.get("data") or {}, path="data"))
    val = ValidationConfig(**_pick_dataclass_kwargs(ValidationConfig, raw.get("validation") or {}, path="validation"))
    out = OutputConfig(**_pick_dataclass_kwargs(OutputConfig, raw.get("output") or {}, path="output"))
    model = model_config_from_dict(raw.get("model") or {})
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        data=data,
        features=dict(raw.get("features") or {}),
        labeling=dict(raw.get("labeling") or {}),
        model=model,
        validation=val,
        walk_forward_calibration=walk_forward_calibration_from_dict(
            raw.get("walk_forward_calibration") if isinstance(raw.get("walk_forward_calibration"), dict) else None
        ),
        backtest=dict(raw.get("backtest") or {}),
        output=out,
        metadata=dict(raw.get("metadata") or {}),
        audit=AuditConfig(**_pick_dataclass_kwargs(AuditConfig, raw.get("audit") or {}, path="audit")),
        regime=regime_config_from_dict(raw.get("regime")),
        tracking=tracking_config_from_dict(raw.get("tracking")),
        live_ops=live_ops_config_from_dict(raw.get("live_ops") or raw.get("live")),
        execution=execution_config_from_dict(raw.get("execution")),
        source_path=source_path,
    )
