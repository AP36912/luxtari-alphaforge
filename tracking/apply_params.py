"""
Apply dotted-path hyperparameter updates to :class:`ExperimentConfig` (immutable replace).

Used by grid search; only mutates explicit paths — validation remains pipeline-driven (purged CV).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from config.experiment_schema import ExperimentConfig
from regime.config import RegimeConfig


def apply_param_path(cfg: ExperimentConfig, dotted_key: str, value: Any) -> ExperimentConfig:
    """
    Supported patterns (examples):

    - ``features.rsi_period`` — feature override key (may contain dots in key via ``features.a.b``).
    - ``labeling.triple_barrier.max_bars`` — nested under ``labeling["triple_barrier"]``.
    - ``model.extra_params.n_estimators`` — model kwargs.
    - ``model.name`` — registry model name string.
    - ``validation.train_fraction`` / ``validation.n_splits`` / ``validation.embargo`` / ``validation.label_horizon``.
    - ``regime.filters.enabled`` (bool), ``regime.filters.apply_to_signals``, ``regime.filters.require_both``.
    - ``regime.filters.allowed_sessions`` — pass list or tuple (stored as tuple).
    - ``regime.filters.allowed_volatility_states``.
    - ``regime.enabled``, ``regime.detection_method``, ``regime.add_feature_columns``, ``regime.include_segment_metrics``.
    """
    parts = dotted_key.split(".")
    if not parts:
        raise ValueError("empty param path")

    if parts[0] == "features":
        sub = ".".join(parts[1:]) if len(parts) > 1 else ""
        if not sub:
            raise ValueError("features.* requires a key after 'features.'")
        fe = dict(cfg.features)
        fe[sub] = value
        return replace(cfg, features=fe)

    if parts[0] == "labeling":
        if len(parts) < 2:
            raise ValueError("labeling.* requires at least labeling.KEY")
        lab = dict(cfg.labeling)
        if parts[1] == "triple_barrier":
            if len(parts) < 3:
                raise ValueError("labeling.triple_barrier.KEY required")
            tb = dict(lab.get("triple_barrier") or {})
            tb[parts[2]] = value
            lab["triple_barrier"] = tb
        else:
            k = ".".join(parts[1:])
            lab[k] = value
        return replace(cfg, labeling=lab)

    if parts[0] == "model":
        if len(parts) < 2:
            raise ValueError("model.name or model.extra_params.* required")
        if parts[1] == "name":
            return replace(cfg, model=replace(cfg.model, name=str(value)))
        if parts[1] == "extra_params":
            if len(parts) < 3:
                raise ValueError("model.extra_params.KEY required")
            ep = dict(cfg.model.extra_params)
            ep[parts[2]] = value
            return replace(cfg, model=replace(cfg.model, extra_params=ep))
        raise ValueError(f"unknown model path: {dotted_key}")

    if parts[0] == "validation":
        if len(parts) != 2:
            raise ValueError("validation.FIELD with single field name")
        field = parts[1]
        vc = cfg.validation
        if not hasattr(vc, field):
            raise ValueError(f"unknown validation field: {field}")
        return replace(cfg, validation=replace(vc, **{field: value}))

    if parts[0] == "regime":
        return replace(cfg, regime=_apply_regime_path(cfg.regime, parts[1:], value))

    raise ValueError(f"unsupported param path root: {parts[0]!r}")


def _apply_regime_path(reg: RegimeConfig, parts: list[str], value: Any) -> RegimeConfig:
    if not parts:
        raise ValueError("regime.* requires subpath")
    head, *rest = parts
    if head == "detection" and rest:
        field = rest[0]
        det = reg.detection
        if not hasattr(det, field):
            raise ValueError(f"unknown regime.detection field: {field}")
        return replace(reg, detection=replace(det, **{field: value}))
    if head == "filters" and rest:
        field = rest[0]
        fl = reg.filters
        if field in ("allowed_sessions", "allowed_volatility_states"):
            tup = tuple(value) if isinstance(value, (list, tuple)) else (value,)
            return replace(reg, filters=replace(fl, **{field: tup}))
        if field in ("enabled", "apply_to_signals", "require_both"):
            return replace(reg, filters=replace(fl, **{field: bool(value)}))
        raise ValueError(f"unknown regime.filters field: {field}")
    if head == "enabled":
        return replace(reg, enabled=bool(value))
    if head == "detection_method":
        return replace(reg, detection_method=str(value))  # type: ignore[arg-type]
    if head == "add_feature_columns":
        return replace(reg, add_feature_columns=bool(value))
    if head == "include_segment_metrics":
        return replace(reg, include_segment_metrics=bool(value))
    raise ValueError(f"unknown regime path: regime.{'.'.join(parts)}")
