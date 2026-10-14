import numpy as np
import pytest

from vecsearch.bench import make_clusters, recall_at_k
from vecsearch.core import Metric, prepare, smallest_k, squared_distances, to_distance
from vecsearch.index.flat import FlatIndex
from vecsearch.index.ivf import IVFIndex
from vecsearch.index.kmeans import kmeans
from vecsearch.index.lsh import LSHIndex
from vecsearch.index.registry import create_index


def _ids(n: int) -> list[str]:
    return [str(i) for i in range(n)]


def _truth(
    data: np.ndarray, queries: np.ndarray, k: int, metric: Metric = Metric.L2
) -> list[list[str]]:
    flat = FlatIndex(metric)
    flat.build(_ids(len(data)), data)
    return [[h.id for h in flat.search(q, k)] for q in queries]


def test_smallest_k_orders_and_breaks_ties_by_index() -> None:
    values = np.array([3.0, 1.0, 1.0, 2.0, 1.0])
    assert smallest_k(values, 2).tolist() == [1, 2]
    assert smallest_k(values, 10).tolist() == [1, 2, 4, 3, 0]
    assert smallest_k(values, 0).tolist() == []


def test_prepare_validates_and_normalises() -> None:
    with pytest.raises(ValueError):
        prepare(np.array([1.0, 2.0]), Metric.L2)
    with pytest.raises(ValueError):
        prepare(np.array([[1.0, float("nan")]]), Metric.L2)
    with pytest.raises(ValueError):
        prepare(np.array([[0.0, 0.0]]), Metric.COSINE)
    unit = prepare(np.array([[3.0, 4.0]]), Metric.COSINE)
    assert np.linalg.norm(unit[0]) == pytest.approx(1.0)


def test_flat_matches_numpy_brute_force() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 8))
    q = rng.normal(size=8)
    index = FlatIndex()
    index.build(_ids(200), data)
    expected = np.argsort(((data - q) ** 2).sum(axis=1))[:5]
    assert [int(h.id) for h in index.search(q, 5)] == expected.tolist()


def test_cosine_distance_values() -> None:
    index = FlatIndex(Metric.COSINE)
    index.build(["same", "orthogonal", "opposite"], np.array([[2.0, 0.0], [0.0, 5.0], [-1.0, 0.0]]))
    hits = {h.id: h.distance for h in index.search(np.array([1.0, 0.0]), 3)}
    assert hits["same"] == pytest.approx(0.0)
    assert hits["orthogonal"] == pytest.approx(1.0)
    assert hits["opposite"] == pytest.approx(2.0)
    assert to_distance(2.0, Metric.COSINE) == 1.0


def test_squared_distances() -> None:
    assert squared_distances(np.array([0.0, 0.0]), np.array([[3.0, 4.0]])).tolist() == [25.0]


def test_kmeans_separates_blobs() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(loc=-10, scale=0.1, size=(30, 2))
    b = rng.normal(loc=10, scale=0.1, size=(30, 2))
    _, labels = kmeans(np.vstack([a, b]), 2, seed=0)
    assert len(set(labels[:30])) == 1
    assert len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_ivf_with_all_cells_probed_is_exact() -> None:
    data = make_clusters(500, 16, seed=2)
    queries = make_clusters(20, 16, seed=3)
    index = IVFIndex(nlist=8, nprobe=8)
    index.build(_ids(500), data)
    found = [[h.id for h in index.search(q, 10)] for q in queries]
    assert recall_at_k(_truth(data, queries, 10), found) == 1.0
    assert index.scanned_fraction() == pytest.approx(1.0)


def test_ivf_recall_grows_with_nprobe_and_scans_less_than_everything() -> None:
    data = make_clusters(3000, 32, n_clusters=30, spread=1.0, seed=4)
    queries = make_clusters(60, 32, n_clusters=30, spread=1.0, seed=5)
    truth = _truth(data, queries, 10)
    recalls: list[float] = []
    for nprobe in (1, 4, 16):
        index = IVFIndex(nlist=30, nprobe=nprobe)
        index.build(_ids(len(data)), data)
        recalls.append(recall_at_k(truth, [[h.id for h in index.search(q, 10)] for q in queries]))
        if nprobe == 1:
            assert index.scanned_fraction() < 0.3
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[2] > 0.9


def test_lsh_finds_a_planted_near_duplicate() -> None:
    rng = np.random.default_rng(6)
    data = rng.normal(size=(2000, 32))
    target = 123
    query = data[target] + rng.normal(scale=0.01, size=32)
    index = LSHIndex(Metric.COSINE, n_tables=12, n_bits=10)
    index.build(_ids(2000), data)
    hits = index.search(query, 1)
    assert hits and hits[0].id == str(target)
    assert index.scanned_fraction() < 0.8


def test_lsh_recall_on_clustered_data() -> None:
    data = make_clusters(2000, 32, n_clusters=25, spread=0.6, seed=7)
    queries = make_clusters(50, 32, n_clusters=25, spread=0.6, seed=8)
    truth = _truth(data, queries, 10, Metric.COSINE)
    index = LSHIndex(Metric.COSINE, n_tables=16, n_bits=8)
    index.build(_ids(len(data)), data)
    found = [[h.id for h in index.search(q, 10)] for q in queries]
    assert recall_at_k(truth, found) > 0.5


def test_build_and_search_validation() -> None:
    index = FlatIndex()
    with pytest.raises(ValueError, match="unique"):
        index.build(["a", "a"], np.zeros((2, 2)) + 1)
    with pytest.raises(ValueError, match="number of ids"):
        index.build(["a"], np.ones((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        index.search(np.ones(2))
    index.build(["a", "b"], np.ones((2, 3)))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(2))
    with pytest.raises(ValueError):
        index.search(np.ones(3), k=0)


def test_registry_rejects_unknown_algorithm_and_params() -> None:
    assert create_index("ivf", nlist=4).algorithm == "ivf"
    with pytest.raises(ValueError, match="unknown algorithm"):
        create_index("hnsw")
    with pytest.raises(ValueError, match="unknown parameter"):
        create_index("flat", nlist=4)
    with pytest.raises(ValueError):
        create_index("lsh", n_bits=99)
