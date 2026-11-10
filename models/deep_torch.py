"""
Light PyTorch sequence classifiers: LSTM and 1D CNN.

Expects 2D ``X`` at ``fit``/``predict`` time; internally builds sliding windows
of length ``seq_len``. The first ``seq_len - 1`` rows get ``nan`` predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.artifact_trust import get_model_artifact_trust_policy, trusted_model_dir
from models.base import BaseQuantModel
from models.persistence import model_dir, read_meta, write_meta


def _require_trust(path: str | Path):
    """Yield a verified snapshot for direct or registry-dispatched loads."""
    return trusted_model_dir(path, get_model_artifact_trust_policy())


def _sliding_windows(X: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X_windows`` (n_w, seq_len, n_features) and target row indices."""
    n, f = X.shape
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if n < seq_len:
        return np.empty((0, seq_len, f), dtype=np.float32), np.arange(0, dtype=int)
    windows = np.stack([X[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, n)], axis=0)
    row_idx = np.arange(seq_len - 1, n, dtype=int)
    return windows.astype(np.float32), row_idx


class _TorchSequenceClassifier(BaseQuantModel):
    supports_proba = True
    model_type = "lstm"

    def __init__(
        self,
        *,
        seq_len: int = 16,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 15,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 42,
        _state_payload: dict[str, Any] | None = None,
    ) -> None:
        import torch

        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self._torch = torch
        self._n_features: int | None = None
        self._n_classes: int | None = None
        self._classes: np.ndarray | None = None
        self._net: Any = None
        self._device = torch.device("cpu")

        if _state_payload is not None:
            self._n_features = int(_state_payload["n_features"])
            self._n_classes = int(_state_payload["n_classes"])
            self._classes = np.asarray(_state_payload["classes"])
            self._build_net()
            self._net.load_state_dict(_state_payload["state_dict"])
            self._net.to(self._device)

    def _init_kwargs(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }

    def _build_net(self) -> None:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseQuantModel:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        self._classes = np.unique(y)
        self._n_classes = int(len(self._classes))
        self._n_features = int(X.shape[1])
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_int = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        Xw, row_idx = _sliding_windows(X, self.seq_len)
        if len(Xw) == 0:
            raise ValueError("Not enough rows for seq_len")
        yw = y_int[row_idx]

        self._build_net()
        assert self._net is not None
        self._net.to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        ds = TensorDataset(
            torch.from_numpy(Xw),
            torch.from_numpy(yw),
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                opt.zero_grad()
                logits = self._forward_net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        self._net.eval()
        return self

    def _forward_net(self, xb: Any) -> Any:
        return self._net(xb)

    def _windowed_forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        Xw, row_idx = _sliding_windows(X, self.seq_len)
        if len(Xw) == 0 or self._net is None:
            return np.empty((0, self._n_classes or 1)), row_idx
        import torch

        self._net.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(Xw), self.batch_size):
                batch = torch.from_numpy(Xw[i : i + self.batch_size]).to(self._device)
                logits = self._forward_net(batch)
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                outs.append(prob)
        return np.vstack(outs), row_idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None or self._classes is None or self._n_classes is None:
            raise RuntimeError("Model is not fitted")
        probs, row_idx = self._windowed_forward(X)
        n = len(X)
        out = np.full((n, self._n_classes), np.nan, dtype=np.float64)
        out[row_idx] = probs
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        out = np.full(len(X), np.nan, dtype=np.float64)
        mask = ~np.isnan(proba).all(axis=1)
        if not mask.any():
            return out
        cls_idx = np.argmax(proba[mask], axis=1)
        cl = self._classes
        assert cl is not None
        out[mask] = cl[cls_idx.astype(int)]
        return out

    @property
    def classes_(self) -> np.ndarray | None:
        return self._classes

    def save(self, path: str | Path) -> None:
        if self._net is None or self._classes is None:
            raise RuntimeError("Cannot save an unfitted model")
        import torch

        d = model_dir(path)
        payload = {
            "state_dict": {k: v.cpu() for k, v in self._net.state_dict().items()},
            "n_features": self._n_features,
            "n_classes": self._n_classes,
            "classes": self._classes.tolist(),
            "init_kwargs": self._init_kwargs(),
        }
        torch.save(payload, d / "torch.pt")
        write_meta(d, {"model_type": self.model_type, "init_kwargs": self._init_kwargs()})

    @classmethod
    def load(cls, path: str | Path) -> BaseQuantModel:
        import torch

        with _require_trust(path) as d:
            meta = read_meta(d)
            if meta.get("model_type") != cls.model_type:
                raise ValueError(
                    f"Expected model_type {cls.model_type!r}, got {meta.get('model_type')!r}"
                )
            try:
                payload = torch.load(d / "torch.pt", map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise RuntimeError(
                    "Safe PyTorch model loading requires torch.load(..., weights_only=True); "
                    "upgrade PyTorch before loading this artifact."
                ) from exc
            init_kw = {
                k: v
                for k, v in meta.get("init_kwargs", {}).items()
                if k != "_state_payload"
            }
            return cls(_state_payload=payload, **init_kw)


class LSTMClassifierModel(_TorchSequenceClassifier):
    model_type = "lstm"

    def _build_net(self) -> None:
        import torch.nn as nn

        if self._n_features is None or self._n_classes is None:
            return

        nf = self._n_features
        h = self.hidden_size
        nl = self.num_layers
        dr = self.dropout
        nc = self._n_classes

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    nf,
                    h,
                    nl,
                    batch_first=True,
                    dropout=dr if nl > 1 else 0.0,
                )
                self.fc = nn.Linear(h, nc)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._net = Net()


class CNN1DClassifierModel(_TorchSequenceClassifier):
    model_type = "cnn"

    def _build_net(self) -> None:
        import torch.nn as nn

        if self._n_features is None or self._n_classes is None:
            return

        nf = self._n_features
        nc = self._n_classes
        hs = self.hidden_size

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(nf, hs, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hs, hs, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Linear(hs, nc)

            def forward(self, x):
                x = x.transpose(1, 2)
                h = self.conv(x).squeeze(-1)
                return self.fc(h)

        self._net = Net()
