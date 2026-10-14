"""Persist an index as plain files (no pickle): ``vectors.npy`` plus ``manifest.json``.

Only the raw vectors, ids and construction parameters are stored. The search structures are
rebuilt on load, which is deterministic because every random component is seeded. That keeps the
files safe to load from untrusted locations and independent of the internal data layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vecsearch.core import Metric
from vecsearch.index.base import Index
from vecsearch.index.registry import create_index


def save_index(index: Index, directory: Path) -> None:
    if len(index) == 0:
        raise ValueError("cannot save an empty index")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", index._data, allow_pickle=False)
    manifest = {
        "algorithm": index.algorithm,
        "metric": index.metric.value,
        "params": index.params,
        "ids": index._ids,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def load_index(directory: Path) -> Index:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    index = create_index(manifest["algorithm"], Metric(manifest["metric"]), **manifest["params"])
    index.build(manifest["ids"], vectors)
    return index
