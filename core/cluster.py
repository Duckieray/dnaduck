from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from .utils import get_logger

LOGGER = get_logger(__name__)


def resolve_eps(config: dict) -> float:
    mode = str(config.get("mode", "realism")).lower().strip()
    eps_realism = float(config.get("eps_realism", 0.60))
    eps_anime = float(config.get("eps_anime", 0.75))
    eps_hybrid = float(config.get("eps_hybrid", (eps_realism + eps_anime) / 2.0))

    if mode == "realism":
        return eps_realism
    if mode == "anime":
        return eps_anime
    if mode == "hybrid":
        return eps_hybrid
    raise ValueError(f"Unsupported mode '{mode}'. Expected realism | anime | hybrid.")


def cluster_faces(embeddings: np.ndarray, config: dict) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=int)

    eps = resolve_eps(config)
    min_samples = int(config.get("min_samples", 3))
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    clustered_count = int(np.sum(labels != -1))
    noise_count = int(np.sum(labels == -1))
    cluster_count = len({int(v) for v in labels.tolist() if int(v) != -1})
    LOGGER.info(
        "Clustering complete: mode=%s eps=%.3f min_samples=%d clusters=%d clustered=%d noise=%d",
        str(config.get("mode", "realism")),
        eps,
        min_samples,
        cluster_count,
        clustered_count,
        noise_count,
    )
    return labels.astype(int)

