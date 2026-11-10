"""Reproducibility helpers and environment snapshots."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any


def environment_snapshot() -> dict[str, Any]:
    """Versions useful for audit trails (not a full conda export)."""
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import numpy as np

        out["numpy"] = np.__version__
    except ImportError:
        out["numpy"] = None
    try:
        import pandas as pd

        out["pandas"] = pd.__version__
    except ImportError:
        out["pandas"] = None
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except ImportError:
        out["sklearn"] = None
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError:
        out["torch"] = None
    return out


def fingerprint_json_blob(obj: Any) -> str:
    """Stable short hash of a JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def assert_numpy_determinism_after_seed(seed: int, *, trials: int = 2) -> bool:
    """Return True if ``np.random.randn`` matches across two resets (sanity check)."""
    import numpy as np

    outs: list[str] = []
    for _ in range(trials):
        np.random.seed(seed)
        outs.append(np.random.randn(5).tobytes().hex())
    return len(set(outs)) == 1
