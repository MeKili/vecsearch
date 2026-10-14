"""IVF-Flat: partition the data with k-means, then scan only the closest ``nprobe`` cells."""

from __future__ import annotations

import numpy as np

from vecsearch.core import FloatArray, IntArray, Metric, squared_distances
from vecsearch.index.base import Index
from vecsearch.index.kmeans import kmeans


class IVFIndex(Index):
    algorithm = "ivf"

    def __init__(
        self, metric: Metric = Metric.L2, nlist: int = 16, nprobe: int = 2, seed: int = 0
    ) -> None:
        super().__init__(metric)
        if nlist < 1 or nprobe < 1:
            raise ValueError("nlist and nprobe must be >= 1")
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed
        self._centroids: FloatArray = np.empty((0, 0), dtype=np.float64)
        self._cells: list[IntArray] = []

    @property
    def params(self) -> dict[str, int]:
        return {"nlist": self.nlist, "nprobe": self.nprobe, "seed": self.seed}

    def _build(self) -> None:
        self._centroids, assignments = kmeans(self._data, self.nlist, seed=self.seed)
        self._cells = [
            np.flatnonzero(assignments == c).astype(np.int64) for c in range(len(self._centroids))
        ]

    def _candidates(self, query: FloatArray) -> IntArray:
        nearest = np.argsort(squared_distances(query, self._centroids))[: self.nprobe]
        return np.concatenate([self._cells[int(c)] for c in nearest])
