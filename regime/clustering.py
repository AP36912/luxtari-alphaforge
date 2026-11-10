"""
Optional offline regime clustering (K-means) on causal regime features.

Fit on a **prefix** of rows (e.g. training sample) and assign labels to all rows — suitable
for research; live use would refit on a rolling window outside this helper.
"""

from __future__ import annotations

import numpy as np


def fit_predict_regime_clusters(
    features: np.ndarray,
    *,
    fit_end_index: int,
    n_clusters: int = 4,
    random_state: int = 0,
    min_train_rows: int = 50,
) -> np.ndarray:
    """
    Fit K-means on ``features[:fit_end_index]`` and predict labels for every row.

    Rows before ``min_train_rows`` valid samples are labeled ``-1``. Invalid fit (too few rows)
    returns all ``-1``.

    Args:
        features: 2D array ``(n_rows, n_features)``.
        fit_end_index: Exclusive end index of training slice (e.g. chronological split).
    """
    n = features.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if fit_end_index < min_train_rows or n_clusters < 2:
        return out
    X_tr = np.asarray(features[:fit_end_index], dtype=np.float64)
    if np.any(~np.isfinite(X_tr)):
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    if X_tr.shape[0] < min_train_rows:
        return out
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cluster_offline requires scikit-learn (sklearn)") from exc

    km = KMeans(
        n_clusters=n_clusters,
        random_state=int(random_state),
        n_init=10,
    )
    km.fit(X_tr)
    X_all = np.asarray(features, dtype=np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    return km.predict(X_all).astype(np.int64)
