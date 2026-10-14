"""Benchmark harness: recall against exact search, latency and how much data each query touches."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vecsearch.core import FloatArray, Metric
from vecsearch.index.base import Index
from vecsearch.index.flat import FlatIndex


def make_clusters(
    n: int, dim: int, n_clusters: int = 20, spread: float = 0.5, seed: int = 0
) -> FloatArray:
    """Gaussian blobs around random centres. Real embeddings cluster like this, uniform noise
    does not, and the difference decides how well IVF and LSH work."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    data: FloatArray = centres[labels] + rng.normal(scale=spread, size=(n, dim))
    return data


@dataclass(frozen=True)
class BenchRow:
    label: str
    build_s: float
    recall: float
    qps: float
    p50_ms: float
    p95_ms: float
    scanned: float


def recall_at_k(truth: Sequence[Sequence[str]], found: Sequence[Sequence[str]]) -> float:
    """Average share of the true top-k ids that the approximate search also returned."""
    if len(truth) != len(found) or not truth:
        raise ValueError("truth and found must be non-empty and the same length")
    scores = [len(set(t) & set(f)) / len(t) for t, f in zip(truth, found, strict=True) if t]
    return float(np.mean(scores))


def evaluate(
    label: str, index: Index, queries: FloatArray, truth: Sequence[Sequence[str]], k: int
) -> BenchRow:
    index.queries = 0
    index.candidates_scored = 0
    latencies: list[float] = []
    found: list[list[str]] = []
    for q in queries:
        start = time.perf_counter()
        hits = index.search(q, k)
        latencies.append(time.perf_counter() - start)
        found.append([h.id for h in hits])
    total = sum(latencies)
    return BenchRow(
        label=label,
        build_s=0.0,
        recall=recall_at_k(truth, found),
        qps=len(latencies) / total if total > 0 else float("inf"),
        p50_ms=float(np.percentile(latencies, 50) * 1000),
        p95_ms=float(np.percentile(latencies, 95) * 1000),
        scanned=index.scanned_fraction(),
    )


def run_benchmark(
    indexes: Sequence[tuple[str, Index]],
    data: FloatArray,
    queries: FloatArray,
    k: int = 10,
    metric: Metric = Metric.L2,
) -> list[BenchRow]:
    """Build every index on ``data`` and score it against exact search on ``queries``."""
    ids = [str(i) for i in range(data.shape[0])]
    exact = FlatIndex(metric)
    exact.build(ids, data)
    truth = [[h.id for h in exact.search(q, k)] for q in queries]
    rows: list[BenchRow] = []
    for label, index in indexes:
        start = time.perf_counter()
        index.build(ids, data)
        build_s = time.perf_counter() - start
        row = evaluate(label, index, queries, truth, k)
        rows.append(BenchRow(**{**row.__dict__, "build_s": build_s}))
    return rows


def format_table(rows: Sequence[BenchRow]) -> str:
    head = (
        f"{'index':<18} {'recall':>7} {'scanned':>8} {'qps':>9} "
        f"{'p50 ms':>8} {'p95 ms':>8} {'build s':>8}"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.label:<18} {r.recall:>7.3f} {r.scanned:>8.1%} {r.qps:>9.0f} "
            f"{r.p50_ms:>8.3f} {r.p95_ms:>8.3f} {r.build_s:>8.3f}"
        )
    return "\n".join(lines)
