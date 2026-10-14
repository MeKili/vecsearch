"""Create an index from its algorithm name and parameters."""

from __future__ import annotations

from vecsearch.core import Metric
from vecsearch.index.base import Index
from vecsearch.index.flat import FlatIndex
from vecsearch.index.ivf import IVFIndex
from vecsearch.index.lsh import LSHIndex

ALGORITHMS = ("flat", "ivf", "lsh")


def create_index(algorithm: str, metric: Metric = Metric.L2, **params: int) -> Index:
    if algorithm == "flat":
        _reject_params(algorithm, params, ())
        return FlatIndex(metric)
    if algorithm == "ivf":
        _reject_params(algorithm, params, ("nlist", "nprobe", "seed"))
        return IVFIndex(metric, **params)
    if algorithm == "lsh":
        _reject_params(algorithm, params, ("n_tables", "n_bits", "seed"))
        return LSHIndex(metric, **params)
    raise ValueError(f"unknown algorithm {algorithm!r}; choose from {', '.join(ALGORITHMS)}")


def _reject_params(algorithm: str, params: dict[str, int], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"unknown parameter(s) for {algorithm}: {', '.join(unknown)}")
