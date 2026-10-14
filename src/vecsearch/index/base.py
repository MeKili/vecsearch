"""Common machinery: every index is a candidate generator followed by exact re-scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from vecsearch.core import (
    FloatArray,
    Hit,
    IntArray,
    Metric,
    prepare,
    smallest_k,
    squared_distances,
    to_distance,
)


class Index(ABC):
    algorithm = "base"

    def __init__(self, metric: Metric = Metric.L2) -> None:
        self.metric = metric
        self._ids: list[str] = []
        self._data: FloatArray = np.empty((0, 0), dtype=np.float64)
        self.queries = 0
        self.candidates_scored = 0

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return int(self._data.shape[1])

    @property
    def params(self) -> dict[str, int]:
        """Construction parameters needed to rebuild the index from its vectors."""
        return {}

    def build(self, ids: Sequence[str], vectors: FloatArray) -> None:
        """Index ``vectors`` (n x d) under the given unique ids, replacing any previous content."""
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        matrix = prepare(vectors, self.metric)
        if len(ids) != matrix.shape[0]:
            raise ValueError("number of ids must equal number of vectors")
        self._ids = list(ids)
        self._data = matrix
        self._build()
        self.queries = 0
        self.candidates_scored = 0

    @abstractmethod
    def _build(self) -> None:
        """Build auxiliary structures over ``self._data``."""

    @abstractmethod
    def _candidates(self, query: FloatArray) -> IntArray:
        """Row indices worth scoring exactly for this (prepared) query."""

    def search(self, query: FloatArray, k: int = 10) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if len(self) == 0:
            raise ValueError("index is empty")
        q = prepare(np.asarray(query, dtype=np.float64).reshape(1, -1), self.metric)[0]
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dimension {q.shape[0]}, index has {self.dim}")
        rows = self._candidates(q)
        self.queries += 1
        self.candidates_scored += int(rows.size)
        if rows.size == 0:
            return []
        dists = squared_distances(q, self._data[rows])
        best = smallest_k(dists, k)
        return [
            Hit(self._ids[int(rows[i])], to_distance(float(dists[i]), self.metric)) for i in best
        ]

    def scanned_fraction(self) -> float:
        """Average share of the dataset scored exactly per query so far."""
        if self.queries == 0 or len(self) == 0:
            return 0.0
        return self.candidates_scored / (self.queries * len(self))
