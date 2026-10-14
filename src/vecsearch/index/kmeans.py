"""Lloyd's k-means with k-means++ seeding, used to partition data for the IVF index."""

from __future__ import annotations

import numpy as np

from vecsearch.core import FloatArray, IntArray


def _assign(data: FloatArray, centroids: FloatArray) -> IntArray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, and ||x||^2 does not affect the argmin
    scores = (centroids**2).sum(axis=1)[None, :] - 2.0 * data @ centroids.T
    out: IntArray = np.argmin(scores, axis=1).astype(np.int64)
    return out


def kmeans(
    data: FloatArray, k: int, iterations: int = 20, seed: int = 0
) -> tuple[FloatArray, IntArray]:
    """Return (centroids, assignments). ``k`` is capped at the number of points."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = data.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)

    centroids = np.empty((k, data.shape[1]), dtype=np.float64)
    centroids[0] = data[rng.integers(n)]
    closest = ((data - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        probs = closest / total if total > 0 else np.full(n, 1.0 / n)
        centroids[i] = data[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((data - centroids[i]) ** 2).sum(axis=1))

    assignments = _assign(data, centroids)
    for _ in range(iterations):
        for j in range(k):
            members = data[assignments == j]
            if members.shape[0] > 0:
                centroids[j] = members.mean(axis=0)
        updated = _assign(data, centroids)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
    return centroids, assignments
