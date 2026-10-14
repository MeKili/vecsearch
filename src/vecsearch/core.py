"""Shared types and numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Metric(StrEnum):
    L2 = "l2"
    COSINE = "cosine"


@dataclass(frozen=True)
class Hit:
    """A search result. For cosine the distance is ``1 - cosine similarity``."""

    id: str
    distance: float


def prepare(vectors: FloatArray, metric: Metric) -> FloatArray:
    """Validate a 2-D matrix and, for cosine, scale each row to unit length.

    After this step every index works in squared Euclidean space: for unit vectors the squared
    distance is exactly ``2 * (1 - cosine)``, so one code path serves both metrics.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain NaN or infinite values")
    if metric is Metric.COSINE:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("cosine distance is undefined for zero vectors")
        matrix = matrix / norms
    return matrix


def squared_distances(query: FloatArray, matrix: FloatArray) -> FloatArray:
    """Squared Euclidean distance from ``query`` to every row of ``matrix``."""
    diff = matrix - query
    out: FloatArray = np.einsum("ij,ij->i", diff, diff)
    return out


def smallest_k(values: FloatArray, k: int) -> IntArray:
    """Indices of the ``k`` smallest values, ascending, ties broken by lower index."""
    k = min(k, values.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < values.size:
        part = np.argpartition(values, k - 1)[:k]
        cutoff = values[part].max()
        # include every index tied at the cutoff so tie-breaking by index is deterministic
        pool = np.flatnonzero(values <= cutoff)
    else:
        pool = np.arange(values.size)
    order = np.lexsort((pool, values[pool]))
    return pool[order][:k].astype(np.int64)


def to_distance(squared: float, metric: Metric) -> float:
    """Convert a squared Euclidean distance on prepared vectors to the user-facing distance."""
    if metric is Metric.COSINE:
        return squared / 2.0
    return float(np.sqrt(squared))
