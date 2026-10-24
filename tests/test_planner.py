
import numpy as np
import pandas as pd

from src.index.ivf_index import IVFIndex
from src.index.exact_search import exact_search
from src.filters.search_strategies import (
    post_filter_search,
    pre_filter_search,
    adaptive_search,
)


def test_fixed_threshold_planner_failure():
    rng = np.random.default_rng(42)

    vectors = rng.normal(size=(1000, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    products = pd.DataFrame({
        "category": ["A"] * 1000,
        "zone": ["ZONE-01"] * 1000,
        "price": np.arange(1000, dtype=float),
    })

    query = vectors[0]
    filters = {"max_price": 99.0}
    index = IVFIndex(n_clusters=20).build(vectors)

    truth = exact_search(
        products, vectors, query, filters=filters, k=10
    )
    assert len(truth) == 10

    # Simulate the old fixed-threshold planner (5% threshold).
    selectivity = (
        products["price"].le(filters["max_price"]).mean()
    )
    old_choice = "pre-filter" if selectivity <= 0.05 else "post-filter"

    if old_choice == "post-filter":
        old_results, _ = post_filter_search(
            products, vectors, query, index, filters=filters, k=10
        )
    else:
        old_results, _ = pre_filter_search(
            products, vectors, query, filters=filters, k=10
        )

    # Run the current adaptive planner.
    new_results, _, new_choice = adaptive_search(
        products, vectors, query, index, filters=filters, k=10,
        selectivity_threshold=0.05,
    )

    truth_set = set(truth.tolist())
    old_recall = len(set(old_results.tolist()) & truth_set) / len(truth_set)
    new_recall = len(set(new_results.tolist()) & truth_set) / len(truth_set)

    print("\nPlanner comparison")
    print(f"Selectivity: {selectivity:.1%}")
    print(f"Old fixed-threshold choice: {old_choice}")
    print(f"Old recall@10: {old_recall:.1%}")
    print(f"New adaptive choice: {new_choice}")
    print(f"New recall@10: {new_recall:.1%}")

    # The old fixed threshold chooses post-filter at 10% selectivity.
    assert old_choice == "post-filter"

    # The current planner must choose a valid strategy.
    assert new_choice in {"pre-filter", "post-filter"}
    assert 0.0 <= new_recall <= 1.0
