"""
Load persisted research models and run inference on live feature vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.base import BaseQuantModel
from models.registry import load_model

from live.live_types import LiveSignal, RegimeSnapshot
from live.regime_service import RegimeService
from live.signal_mapping import (
    Phase21SignalMappingParams,
    apply_phase21_direction_pipeline,
    default_direction_from_class,
)
from live.signal_diagnostics import classify_direction_mapping

logger = logging.getLogger(__name__)


def _confidence_from_proba(proba: np.ndarray | None, pred_class: int) -> float:
    if proba is None or proba.ndim != 2 or proba.shape[1] < 1:
        return 1.0
    row = proba[0]
    return float(np.max(row))


@dataclass
class ModelArtifactRef:
    """On-disk model directory; loading requires an active trusted-artifact policy."""

    path: Path
    meta: dict[str, Any] | None = None


@dataclass
class LiveSignalServiceConfig:
    primary: ModelArtifactRef
    feature_column_order: list[str]
    regime_service: RegimeService
    symbol: str = "SYNTH"
    class_to_direction: dict[int, int] | None = None
    meta_model: ModelArtifactRef | None = None
    # Phase 21 (optional; defaults preserve legacy behavior)
    signal_min_confidence_for_entry: float | None = None
    signal_min_confidence_for_entry_long: float | None = None
    signal_min_confidence_for_entry_short: float | None = None
    signal_flat_if_max_proba_below: float | None = None
    signal_allow_neutral_class: bool = False
    signal_neutral_lean_threshold: float = 0.0
    signal_class_mapping_override: dict[int, int] | None = None



class SignalService:
    """
    Primary (+ optional meta) inference. Meta model receives the same feature
    vector by default (train a meta model on those features in research). Model
    paths are loaded through the registry and are denied unless the caller first
    configures an explicit artifact trust policy with a verified SHA-256 digest.
    """

    def __init__(self, cfg: LiveSignalServiceConfig) -> None:
        self._cfg = cfg
        self._primary: BaseQuantModel = load_model(cfg.primary.path)
        self._meta: BaseQuantModel | None = (
            load_model(cfg.meta_model.path) if cfg.meta_model is not None else None
        )
        self._sr_long_completed = 0
        self._sr_short_completed = 0
        self._sr_skipped_none = 0
        self._sr_skipped_conflict = 0
        self._sr_skipped_no_buffer = 0
        self._model_calls = 0
        self._sr_gate_pass = 0
        self._sr_gate_skip = 0
        self._sr_lookback_allow = 0
        self._sr_lookback_skip = 0


    @property
    def primary_model(self) -> BaseQuantModel:
        return self._primary

    @property
    def meta_model(self) -> BaseQuantModel | None:
        return self._meta

    @property
    def regime_service(self) -> RegimeService:
        return self._cfg.regime_service

    def strategy_setup_counters_snapshot(self) -> dict[str, Any]:
        """Compatibility diagnostics: proprietary setup gating is unavailable publicly."""
        return {
            "strategy_setup_gating_enabled": False,
            "strategy_setup_lookback_bars": 0,
            "n_model_inference_calls": int(self._model_calls),
        }


    def generate(
        self,
        X: np.ndarray,
        *,
        row: pd.Series,
        regime_snapshot: RegimeSnapshot | None = None,
        ohlcv_buffer: pd.DataFrame | None = None,
        strategy_gate_bypass: bool = False,
    ) -> LiveSignal:
        sig, _diag = self.generate_with_diagnostics(
            X,
            row=row,
            regime_snapshot=regime_snapshot,
            ohlcv_buffer=ohlcv_buffer,
            strategy_gate_bypass=strategy_gate_bypass,
        )
        if sig is None:
            raise RuntimeError(
                "LiveSignal is None (strategy setup gate skipped inference). "
                "Use generate_with_diagnostics(..., ohlcv_buffer=...) for diagnostics."
            )
        return sig

    def generate_with_diagnostics(
        self,
        X: np.ndarray,
        *,
        row: pd.Series,
        regime_snapshot: RegimeSnapshot | None = None,
        ohlcv_buffer: pd.DataFrame | None = None,
        strategy_gate_bypass: bool = False,
    ) -> tuple[LiveSignal | None, dict[str, Any]]:
        """
        Run primary (+ optional meta) inference and return a structured explanation of direction mapping.

        Meta model output is **diagnostic only**; it does not alter ``LiveSignal.direction`` today.

        When ``strategy_setup_gating_enabled`` and a config are set, returns ``(None, diag)`` if the
        last bar of ``ohlcv_buffer`` is not a strategy setup-completion bar (no model call).
        """
        regime = regime_snapshot or self._cfg.regime_service.snapshot_from_row(row)
        ts = pd.Timestamp(row["timestamp"])

        # Public inference has no proprietary strategy-setup gate.
        setup_kind_pass: str | None = None
        setup_gate_was_checked = False

        self._model_calls += 1
        if setup_gate_was_checked:
            self._sr_gate_pass += 1
        pred = int(np.asarray(self._primary.predict(X)).reshape(-1)[0])
        proba = None
        raw_list: list[float] | None = None
        proba_error: str | None = None
        if self._primary.supports_proba:
            try:
                proba = np.asarray(self._primary.predict_proba(X))
                raw_list = [float(x) for x in proba.reshape(-1)]
            except Exception as exc:  # noqa: BLE001
                proba = None
                proba_error = str(exc)
        conf = _confidence_from_proba(proba, pred)

        primary_classes = getattr(self._primary, "classes_", None)
        classes_list: list[Any] | None = None
        if primary_classes is not None:
            try:
                classes_list = [int(x) for x in np.asarray(primary_classes).reshape(-1)]
            except Exception:  # noqa: BLE001
                classes_list = [str(x) for x in np.asarray(primary_classes).reshape(-1)]

        p21_params = Phase21SignalMappingParams(
            signal_min_confidence_for_entry=self._cfg.signal_min_confidence_for_entry,
            signal_min_confidence_for_entry_long=self._cfg.signal_min_confidence_for_entry_long,
            signal_min_confidence_for_entry_short=self._cfg.signal_min_confidence_for_entry_short,
            signal_flat_if_max_proba_below=self._cfg.signal_flat_if_max_proba_below,
            signal_allow_neutral_class=self._cfg.signal_allow_neutral_class,
            signal_neutral_lean_threshold=float(self._cfg.signal_neutral_lean_threshold),
            signal_class_mapping_override=self._cfg.signal_class_mapping_override,
        )
        direction, phase21_trace = apply_phase21_direction_pipeline(
            pred=pred,
            proba=proba,
            raw_proba_list=raw_list,
            classes_list=classes_list,
            research_class_to_direction=self._cfg.class_to_direction,
            params=p21_params,
        )
        direction_after_p21 = int(direction)

        max_p: float | None = float(max(raw_list)) if raw_list else None

        entry_pattern_diag: dict[str, Any] = {
            "blocks_trades": False,
            "reference_only": True,
            "skipped": "proprietary_entry_pattern_removed",
            "max_class_probability": max_p,
        }

        direction = direction_after_p21
        final_direction_reason = str(phase21_trace.get("final_direction_reason", ""))

        mapping_kind = str(phase21_trace.get("direction_mapping_rule", ""))
        legacy_kind = classify_direction_mapping(pred, self._cfg.class_to_direction)

        meta_cls: int | None = None
        meta_error: str | None = None
        if self._meta is not None:
            try:
                meta_cls = int(np.asarray(self._meta.predict(X)).reshape(-1)[0])
            except Exception as exc:  # noqa: BLE001
                meta_error = str(exc)

        logits_note: str | None = None
        raw_scores: list[float] | None = None
        dec_fn = getattr(self._primary, "decision_function", None)
        if callable(dec_fn):
            try:
                scores = np.asarray(dec_fn(X)).reshape(-1)
                raw_scores = [float(x) for x in scores.tolist()]
            except Exception as exc:  # noqa: BLE001
                logits_note = f"decision_function_failed:{exc}"
        else:
            logits_note = "primary_model_has_no_decision_function"

        diag: dict[str, Any] = {
            "model_skipped": False,
            "strategy_setup_decision_bar_kind": setup_kind_pass,
            "model_primary_predicted_class": pred,
            "primary_model_classes_": classes_list,
            "primary_supports_proba": bool(self._primary.supports_proba),
            "raw_proba": raw_list,
            "per_class_probability": phase21_trace.get("per_class_probability"),
            "predict_proba_error": proba_error,
            "primary_raw_scores_decision_function": raw_scores,
            "primary_scores_note": logits_note,
            "confidence_computed": conf,
            "confidence_rule": "max_predicted_class_probability"
            if raw_list
            else "no_proba_available_used_1.0",
            "direction_mapping_rule": mapping_kind,
            "direction_mapping_rule_without_execution_override": legacy_kind,
            "class_to_direction_config": self._cfg.class_to_direction,
            "signal_class_mapping_override_config": self._cfg.signal_class_mapping_override,
            "direction_from_model_mapping": phase21_trace.get("direction_after_class_mapping"),
            "direction_after_probability_gates": direction_after_p21,
            "direction_after_entry_filter": int(direction),
            "note_entry_pattern_is_diagnostic_only": True,
            "final_direction_reason": final_direction_reason,
            "entry_filter": entry_pattern_diag,
            "phase21": phase21_trace,
            "meta_model_present": self._meta is not None,
            "meta_predicted_class": meta_cls,
            "meta_predict_error": meta_error,
            "meta_note": (
                "Meta model class is recorded for diagnostics only; "
                "SignalService does not use it to set LiveSignal.direction."
            ),
            "neutral_flat_explanation": phase21_trace.get("neutral_flat_explanation"),
        }

        sig = LiveSignal(
            timestamp=ts,
            symbol=self._cfg.symbol,
            direction=direction,
            confidence=conf,
            regime=regime,
            primary_class=pred,
            meta_class=meta_cls,
            raw_proba=raw_list,
        )
        return sig, diag


# Backward-compatible name for tests / callers
_default_direction_from_class = default_direction_from_class
