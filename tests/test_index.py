
import numpy as np
from src.index.ivf_index import IVFIndex
from src.index.exact_search import exact_search
import pandas as pd

from src.filters.search_strategies import (
    post_filter_search,
    pre_filter_search,
    adaptive_search,
)


def test_index_returns_valid_results():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(500, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = IVFIndex(n_clusters=10).build(vectors)
    results = index.search(vectors[0], k=10, n_probe=3)

    assert len(results) == 10
    assert len(set(results.tolist())) == 10
    assert np.all(results >= 0)
    assert np.all(results < len(vectors))


def test_index_rejects_wrong_query_dimension():
    vectors = np.eye(32, dtype=np.float32)
    index = IVFIndex(n_clusters=4).build(vectors)

    try:
        index.search(np.ones(10, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong query dimension was accepted.")



def test_exact_search_respects_filters():
    import pandas as pd

    products = pd.DataFrame({
        "category": ["A", "B", "A"],
        "zone": ["ZONE-01", "ZONE-01", "ZONE-02"],
        "price": [100.0, 200.0, 150.0],
    })

    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
    ], dtype=np.float32)

    query = np.array([1.0, 0.0], dtype=np.float32)

    result = exact_search(
        products,
        vectors,
        query,
        filters={"category": "A"},
        k=10,
    )

    assert result.tolist() == [0, 2]



def test_search_strategies_respect_filters():
    products = pd.DataFrame({
        "category": ["Luggage", "Books", "Luggage", "Luggage"],
        "zone": ["ZONE-01", "ZONE-01", "ZONE-02", "ZONE-01"],
        "price": [100.0, 200.0, 150.0, 120.0],
    })

    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
    ], dtype=np.float32)

    query = np.array([1.0, 0.0], dtype=np.float32)
    filters = {"category": "Luggage", "zone": "ZONE-01"}

    # Build an IVF index with enough partitions for this small test.
    index = IVFIndex(n_clusters=2).build(vectors)

    post_results, _ = post_filter_search(
        products, vectors, query, index, filters=filters, k=2
    )
    pre_results, _ = pre_filter_search(
        products, vectors, query, filters=filters, k=2
    )
    adaptive_results, _, chosen_strategy = adaptive_search(
        products, vectors, query, index, filters=filters, k=2
    )

    # Only products 0 and 3 satisfy both filters.
    assert set(pre_results.tolist()) == {0, 3}
    assert set(post_results.tolist()).issubset({0, 3})
    assert set(adaptive_results.tolist()).issubset({0, 3})
    assert chosen_strategy == "pre-filter"

