from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vecsearch.api import app
from vecsearch.bench import format_table, make_clusters, recall_at_k, run_benchmark
from vecsearch.cli import main
from vecsearch.core import Metric
from vecsearch.index.ivf import IVFIndex
from vecsearch.index.registry import create_index
from vecsearch.storage import load_index, save_index

client = TestClient(app)


def test_save_and_load_round_trip_gives_identical_results(tmp_path: Path) -> None:
    data = make_clusters(400, 8, seed=1)
    index = IVFIndex(Metric.COSINE, nlist=8, nprobe=3, seed=5)
    index.build([f"v{i}" for i in range(400)], data)
    save_index(index, tmp_path / "idx")
    restored = load_index(tmp_path / "idx")
    q = data[17]
    assert [h.id for h in index.search(q, 5)] == [h.id for h in restored.search(q, 5)]
    assert restored.params == index.params
    assert restored.metric is Metric.COSINE


def test_save_empty_index_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_index(create_index("flat"), tmp_path / "x")


def test_benchmark_rows_and_table() -> None:
    data = make_clusters(600, 16, seed=2)
    queries = make_clusters(20, 16, seed=3)
    rows = run_benchmark(
        [("flat", create_index("flat")), ("ivf", create_index("ivf", nlist=8, nprobe=2))],
        data,
        queries,
        k=5,
    )
    assert rows[0].recall == 1.0
    assert 0.0 < rows[1].scanned < 1.0
    assert "ivf" in format_table(rows)


def test_recall_validation() -> None:
    with pytest.raises(ValueError):
        recall_at_k([["a"]], [])


def test_cli_bench_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--n", "500", "--dim", "8", "--queries", "10", "--clusters", "8"]) == 0
    out = capsys.readouterr().out
    assert "flat" in out and "ivf nprobe=1" in out and "lsh tables=4" in out


def test_api_full_flow() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 4)).tolist()
    ids = [f"item-{i}" for i in range(50)]
    created = client.put(
        "/indexes/demo",
        json={
            "algorithm": "ivf",
            "params": {"nlist": 4, "nprobe": 4},
            "ids": ids,
            "vectors": vectors,
        },
    )
    assert created.status_code == 200
    assert created.json()["size"] == 50

    assert [i["name"] for i in client.get("/indexes").json()] == ["demo"]
    result = client.post("/indexes/demo/search", json={"vector": vectors[7], "k": 3})
    assert result.status_code == 200
    assert result.json()["hits"][0]["id"] == "item-7"

    assert client.delete("/indexes/demo").status_code == 204
    assert client.post("/indexes/demo/search", json={"vector": vectors[7]}).status_code == 404


def test_api_validation_errors() -> None:
    bad_algo = client.put(
        "/indexes/x", json={"algorithm": "nope", "ids": ["a"], "vectors": [[1.0]]}
    )
    assert bad_algo.status_code == 422
    mismatch = client.put("/indexes/x", json={"ids": ["a", "b"], "vectors": [[1.0]]})
    assert mismatch.status_code == 422
    assert client.get("/health").json()["status"] == "ok"
    assert "lsh" in client.get("/algorithms").json()["algorithms"]
    client.put("/indexes/dims", json={"ids": ["a"], "vectors": [[1.0, 2.0]]})
    wrong_dim = client.post("/indexes/dims/search", json={"vector": [1.0]})
    assert wrong_dim.status_code == 422
    client.delete("/indexes/dims")
