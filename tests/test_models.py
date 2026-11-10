"""Model layer: fit/predict/proba, registry, save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.artifact_trust import (
    ModelArtifactTrustPolicy,
    clear_model_artifact_trust_policy,
    compute_dir_sha256,
    set_model_artifact_trust_policy,
)
from models.config import ModelConfig
from models.linear import LogisticRegressionModel, RidgeModel
from models.registry import create_model, list_registered_models, load_model
from models.training import ModelTrainer
from models.trees import RandomForestModel

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _clear_artifact_trust_policy():
    clear_model_artifact_trust_policy()
    yield
    clear_model_artifact_trust_policy()


def _trust_model_dir(path: Path) -> None:
    digest = compute_dir_sha256(path)
    set_model_artifact_trust_policy(
        ModelArtifactTrustPolicy(
            allow_serialized_load=True,
            allowed_sha256=frozenset({digest}),
        )
    )


def _xy_classification(*, n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) * 0.3 > 0).astype(np.int64)
    return X.astype(np.float64), y


def _xy_multiclass(*, n: int = 300, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = np.column_stack([X[:, 0], -X[:, 0] * 0.5, X[:, 1]])
    y = logits.argmax(axis=1) - 1  # {-1, 0, 1}
    return X.astype(np.float64), y.astype(np.int64)


def test_random_forest_fit_predict_proba() -> None:
    X, y = _xy_classification()
    m = RandomForestModel(n_estimators=20, max_depth=4, random_state=0)
    m.fit(X, y)
    pred = m.predict(X[:10])
    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.all((pred == 0) | (pred == 1))


def test_logistic_multiclass_proba() -> None:
    X, y = _xy_multiclass()
    m = LogisticRegressionModel(max_iter=500)
    m.fit(X, y)
    proba = m.predict_proba(X[:20])
    assert proba.shape[1] == 3
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_ridge_regression_no_proba() -> None:
    rng = np.random.default_rng(0)
    X, y = _xy_classification()
    y_cont = X[:, 0] + 0.1 * rng.normal(size=len(X))
    m = RidgeModel(alpha=1.0)
    m.fit(X, y_cont)
    pred = m.predict(X[:5])
    assert pred.shape == (5,)
    with pytest.raises(NotImplementedError):
        m.predict_proba(X[:5])


def test_save_load_sklearn_roundtrip(tmp_path: Path) -> None:
    X, y = _xy_classification()
    m = RandomForestModel(n_estimators=15, random_state=1)
    m.fit(X, y)
    d = tmp_path / "rf_model"
    m.save(d)
    _trust_model_dir(d)
    m2 = RandomForestModel.load(d)
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))


def test_registry_create_and_alias() -> None:
    cfg = ModelConfig(name="rf", extra_params={"n_estimators": 10, "random_state": 0})
    m = create_model(cfg)
    assert isinstance(m, RandomForestModel)
    assert "random_forest" in list_registered_models()


def test_load_model_dispatch(tmp_path: Path) -> None:
    X, y = _xy_classification(n=80)
    m = LogisticRegressionModel(max_iter=300)
    m.fit(X, y)
    d = tmp_path / "log"
    m.save(d)
    _trust_model_dir(d)
    m2 = load_model(d)
    assert isinstance(m2, LogisticRegressionModel)
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))


def test_model_trainer_api(tmp_path: Path) -> None:
    X, y = _xy_classification()
    tr = ModelTrainer(RandomForestModel(n_estimators=12, random_state=2))
    tr.fit(X, y)
    p = tr.predict(X[:7])
    pr = tr.predict_proba(X[:7])
    assert pr.shape[0] == 7
    d = tmp_path / "ttrainer"
    tr.save(d)
    _trust_model_dir(d)
    tr2 = ModelTrainer.load(d)
    np.testing.assert_array_equal(tr.predict(X), tr2.predict(X))


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosting_fit_predict(name: str) -> None:
    X, y = _xy_multiclass()
    m = create_model(ModelConfig(name=name, extra_params={"n_estimators": 20, "random_state": 0}))
    m.fit(X, y)
    proba = m.predict_proba(X[:15])
    assert proba.shape[0] == 15
    assert proba.shape[1] == 3


def test_lstm_sequence_classifier(tmp_path: Path) -> None:
    X, y = _xy_multiclass(n=400)
    m = create_model(
        ModelConfig(
            name="lstm",
            extra_params={"seq_len": 12, "epochs": 8, "hidden_size": 16, "batch_size": 128},
        )
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.isnan(proba[:11]).all()
    assert np.isfinite(proba[20]).all()
    d = tmp_path / "lstm_m"
    m.save(d)
    _trust_model_dir(d)
    m2 = load_model(d)
    # Sequence models need the full series to rebuild windows at each row.
    np.testing.assert_allclose(proba[50:60], m2.predict_proba(X)[50:60], rtol=1e-4, atol=1e-4)


def test_cnn_classifier(tmp_path: Path) -> None:
    X, y = _xy_multiclass(n=350)
    m = create_model(
        ModelConfig(
            name="cnn",
            extra_params={"seq_len": 10, "epochs": 6, "hidden_size": 16, "batch_size": 128},
        )
    )
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(X),)
    d = tmp_path / "cnn_m"
    m.save(d)
    _trust_model_dir(d)
    m2 = load_model(d)
    mask = np.isfinite(pred) & np.isfinite(m2.predict(X))
    np.testing.assert_array_equal(pred[mask], m2.predict(X)[mask])
