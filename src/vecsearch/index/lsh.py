"""Random-hyperplane LSH with several tables and one-bit multi-probe lookups."""

from __future__ import annotations

import numpy as np

from vecsearch.core import FloatArray, IntArray, Metric
from vecsearch.index.base import Index


class LSHIndex(Index):
    algorithm = "lsh"

    def __init__(
        self, metric: Metric = Metric.COSINE, n_tables: int = 8, n_bits: int = 8, seed: int = 0
    ) -> None:
        super().__init__(metric)
        if n_tables < 1 or not 1 <= n_bits <= 62:
            raise ValueError("n_tables must be >= 1 and n_bits must be in [1, 62]")
        self.n_tables, self.n_bits, self.seed = n_tables, n_bits, seed
        self._planes: FloatArray = np.empty((0, 0, 0), dtype=np.float64)
        self._tables: list[dict[int, IntArray]] = []
        self._powers = (1 << np.arange(n_bits, dtype=np.int64)).astype(np.int64)

    @property
    def params(self) -> dict[str, int]:
        return {"n_tables": self.n_tables, "n_bits": self.n_bits, "seed": self.seed}

    def _codes(self, matrix: FloatArray, table: int) -> IntArray:
        bits = (matrix @ self._planes[table].T) > 0.0
        out: IntArray = bits.astype(np.int64) @ self._powers
        return out

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._planes = rng.normal(size=(self.n_tables, self.n_bits, self.dim))
        self._tables = []
        for t in range(self.n_tables):
            codes = self._codes(self._data, t)
            buckets: dict[int, list[int]] = {}
            for row, code in enumerate(codes.tolist()):
                buckets.setdefault(code, []).append(row)
            self._tables.append({c: np.asarray(r, dtype=np.int64) for c, r in buckets.items()})

    def _candidates(self, query: FloatArray) -> IntArray:
        found: list[IntArray] = []
        for t in range(self.n_tables):
            code = int(self._codes(query.reshape(1, -1), t)[0])
            probes = [code] + [code ^ (1 << b) for b in range(self.n_bits)]
            for probe in probes:
                bucket = self._tables[t].get(probe)
                if bucket is not None:
                    found.append(bucket)
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(found))
