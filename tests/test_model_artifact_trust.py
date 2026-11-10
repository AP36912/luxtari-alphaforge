"""Serialized model artifacts are denied unless location and integrity are trusted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import models.linear as linear_module
from models.artifact_trust import (
    ModelArtifactTrustPolicy,
    UntrustedModelArtifactError,
    clear_model_artifact_trust_policy,
    compute_dir_sha256,
    prepare_trusted_model_dir,
    set_model_artifact_trust_policy,
)
from models.linear import LogisticRegressionModel
from models.registry import load_model


@pytest.fixture(autouse=True)
def _default_deny_policy():
    clear_model_artifact_trust_policy()
    yield
    clear_model_artifact_trust_policy()


def _fake_model_dir(tmp_path: Path) -> Path:
    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    (model_path / "meta.json").write_text(
        json.dumps({"model_type": "logistic_regression"}),
        encoding="utf-8",
    )
    (model_path / "estimator.joblib").write_bytes(b"not-a-real-joblib")
    return model_path


def _allow(tmp_path: Path, digest: str) -> ModelArtifactTrustPolicy:
    policy = ModelArtifactTrustPolicy(
        allow_serialized_load=True,
        allowed_sha256=frozenset({digest}),
        allowed_roots=(tmp_path,),
    )
    set_model_artifact_trust_policy(policy)
    return policy


def test_registry_denies_serialized_load_by_default(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)

    with pytest.raises(UntrustedModelArtifactError, match="denied by default"):
        load_model(model_path)


def test_allow_flag_without_approved_hash_is_insufficient(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    set_model_artifact_trust_policy(
        ModelArtifactTrustPolicy(
            allow_serialized_load=True,
            allowed_roots=(tmp_path,),
        )
    )

    with pytest.raises(UntrustedModelArtifactError, match="approved SHA-256"):
        load_model(model_path)


def test_allow_flag_requires_nonempty_allowed_roots(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    digest = compute_dir_sha256(model_path)
    set_model_artifact_trust_policy(
        ModelArtifactTrustPolicy(
            allow_serialized_load=True,
            allowed_sha256=frozenset({digest}),
        )
    )

    with pytest.raises(UntrustedModelArtifactError, match="allowed root"):
        load_model(model_path)


def test_wrong_hash_is_denied(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    _allow(tmp_path, "0" * 64)

    with pytest.raises(UntrustedModelArtifactError, match="not approved"):
        load_model(model_path)


def test_symlink_file_inside_artifact_is_denied(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    digest = compute_dir_sha256(model_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (model_path / "linked.bin").symlink_to(outside)
    _allow(tmp_path, digest)

    with pytest.raises(UntrustedModelArtifactError, match="symlink"):
        load_model(model_path)


def test_trusted_logistic_regression_round_trip(tmp_path: Path) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    original = LogisticRegressionModel(max_iter=200).fit(X, y)
    model_path = tmp_path / "logistic-model"
    original.save(model_path)

    with pytest.raises(UntrustedModelArtifactError):
        load_model(model_path)
    with pytest.raises(UntrustedModelArtifactError):
        LogisticRegressionModel.load(model_path)

    _allow(tmp_path, compute_dir_sha256(model_path))

    loaded = load_model(model_path)
    np.testing.assert_array_equal(original.predict(X), loaded.predict(X))


def test_approved_hash_outside_allowed_root_is_denied(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    digest = compute_dir_sha256(model_path)
    allowed_root = tmp_path / "different-root"
    allowed_root.mkdir()
    set_model_artifact_trust_policy(
        ModelArtifactTrustPolicy(
            allow_serialized_load=True,
            allowed_sha256=frozenset({digest}),
            allowed_roots=(allowed_root,),
        )
    )

    with pytest.raises(UntrustedModelArtifactError, match="outside"):
        load_model(model_path / ".." / model_path.name)


def test_mutation_before_prepare_is_denied_by_old_digest(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    old_digest = compute_dir_sha256(model_path)
    (model_path / "estimator.joblib").write_bytes(b"changed-before-prepare")
    _allow(tmp_path, old_digest)

    with pytest.raises(UntrustedModelArtifactError, match="not approved"):
        load_model(model_path)


def test_prepared_snapshot_survives_original_mutation(tmp_path: Path) -> None:
    model_path = _fake_model_dir(tmp_path)
    policy = _allow(tmp_path, compute_dir_sha256(model_path))

    artifact = prepare_trusted_model_dir(model_path, policy)
    try:
        snapshot_bytes = (artifact.snapshot_path / "estimator.joblib").read_bytes()
        (model_path / "estimator.joblib").write_bytes(b"changed-after-prepare")
        assert (artifact.snapshot_path / "estimator.joblib").read_bytes() == snapshot_bytes
    finally:
        snapshot_path = artifact.snapshot_path
        artifact.cleanup()
    assert not snapshot_path.exists()


def test_registry_deserializes_snapshot_not_mutated_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    original = LogisticRegressionModel(max_iter=200).fit(X, y)
    model_path = tmp_path / "logistic-model"
    original.save(model_path)
    _allow(tmp_path, compute_dir_sha256(model_path))

    real_load = linear_module.joblib.load
    loaded_from: list[Path] = []

    def mutate_original_then_load(snapshot_file: str | Path):
        snapshot_file = Path(snapshot_file)
        loaded_from.append(snapshot_file)
        (model_path / "estimator.joblib").write_bytes(b"substituted")
        return real_load(snapshot_file)

    monkeypatch.setattr(linear_module.joblib, "load", mutate_original_then_load)
    loaded = load_model(model_path)

    assert loaded_from[0].parent != model_path
    assert not loaded_from[0].parent.exists()
    np.testing.assert_array_equal(original.predict(X), loaded.predict(X))
