"""Exact brute-force search. The ground truth every approximate index is measured against."""

from __future__ import annotations

import numpy as np

from vecsearch.core import FloatArray, IntArray
from vecsearch.index.base import Index


class FlatIndex(Index):
    algorithm = "flat"

    def _build(self) -> None:
        self._all: IntArray = np.arange(len(self), dtype=np.int64)

    def _candidates(self, query: FloatArray) -> IntArray:
        return self._all
