"""``vecsearch bench``: compare the indexes on synthetic clustered data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vecsearch.bench import format_table, make_clusters, run_benchmark
from vecsearch.core import Metric
from vecsearch.index.base import Index
from vecsearch.index.registry import create_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vecsearch", description=__doc__)
    parser.add_argument("--n", type=int, default=20_000, help="number of indexed vectors")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=40)
    parser.add_argument("--metric", choices=[m.value for m in Metric], default="l2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    metric = Metric(args.metric)
    data = make_clusters(args.n, args.dim, args.clusters, seed=args.seed)
    queries = make_clusters(args.queries, args.dim, args.clusters, seed=args.seed)
    nlist = max(8, int(args.n**0.5 / 2))
    indexes: list[tuple[str, Index]] = [("flat", create_index("flat", metric))]
    for nprobe in (1, 2, 4, 8):
        indexes.append(
            (f"ivf nprobe={nprobe}", create_index("ivf", metric, nlist=nlist, nprobe=nprobe))
        )
    for tables in (4, 8, 16):
        indexes.append(
            (f"lsh tables={tables}", create_index("lsh", metric, n_tables=tables, n_bits=10))
        )
    print(format_table(run_benchmark(indexes, data, queries, args.k, metric)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
