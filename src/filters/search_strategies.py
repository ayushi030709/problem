
import time
import numpy as np

from src.filters.metadata_filters import apply_filters
from src.index.exact_search import exact_search


def post_filter_search(products, vectors, query, index, filters=None, k=10):
    """Search IVF candidates first, then apply metadata filters."""
    start = time.perf_counter()

    candidates = index.search(
        query,
        k=min(len(products), max(k * 20, k)),
        n_probe=min(5, index.n_clusters),
    )

    mask = apply_filters(products, filters).to_numpy()
    candidates = candidates[mask[candidates]]

    if len(candidates):
        scores = vectors[candidates] @ query
        order = np.argsort(scores)[::-1][:k]
        results = candidates[order]
    else:
        results = np.array([], dtype=int)

    return results, (time.perf_counter() - start) * 1000


def pre_filter_search(products, vectors, query, filters=None, k=10):
    """Filter eligible products first, then rank them exactly."""
    start = time.perf_counter()

    results = exact_search(
        products, vectors, query, filters=filters, k=k
    )

    return results, (time.perf_counter() - start) * 1000


def estimate_search_costs(
    products, vectors, index, filters=None, k=10
):
    """
    Estimate work for the two strategies.

    These are simple operation-count estimates, not calibrated latency
    predictions. Both estimates use embedding dimensions as a proxy for
    similarity-computation work.
    """
    mask = apply_filters(products, filters).to_numpy()
    eligible_count = int(mask.sum())
    total_count = len(products)
    dimensions = int(vectors.shape[1])

    # Exact pre-filtering compares every eligible vector with the query.
    pre_cost = eligible_count * dimensions

    # Approximate post-filtering searches the selected IVF partitions.
    n_probe = min(5, index.n_clusters)
    partition_sizes = [
        len(index.partitions[i]) for i in range(index.n_clusters)
    ]
    average_partition_size = (
        float(np.mean(partition_sizes)) if partition_sizes else 0.0
    )
    estimated_candidates = min(
        total_count,
        int(np.ceil(n_probe * average_partition_size)),
    )

    # Approximate similarity work plus a rough metadata-checking allowance.
    post_cost = estimated_candidates * dimensions + total_count

    return {
        "eligible_count": eligible_count,
        "selectivity": (
            eligible_count / total_count if total_count else 0.0
        ),
        "estimated_candidates": estimated_candidates,
        "pre_filter_cost": int(pre_cost),
        "post_filter_cost": int(post_cost),
    }


def adaptive_search(
    products, vectors, query, index, filters=None, k=10,
    selectivity_threshold=0.15,
):
    """
    Select a strategy using estimated work and a conservative quality rule.

    Selective filters use exact pre-filtering to avoid missing eligible
    neighbors. For broader filters, choose the lower estimated-cost option.
    """
    estimates = estimate_search_costs(
        products, vectors, index, filters=filters, k=k
    )

    selectivity = estimates["selectivity"]

    # Conservative quality safeguard: highly selective searches use exact
    # search because approximate retrieval may return too few valid results.
    if selectivity <= selectivity_threshold:
        strategy = "pre-filter"
        reason = "selective filters: prioritize complete filtered results"

    elif estimates["pre_filter_cost"] <= estimates["post_filter_cost"]:
        strategy = "pre-filter"
        reason = "estimated exact-search work is lower"

    else:
        strategy = "post-filter"
        reason = "estimated IVF-search work is lower"

    if strategy == "pre-filter":
        results, latency_ms = pre_filter_search(
            products, vectors, query, filters=filters, k=k
        )
    else:
        results, latency_ms = post_filter_search(
            products, vectors, query, index, filters=filters, k=k
        )

    # Keep the existing three-value return contract.
    # The planner's explanation is available as a function attribute.
    adaptive_search.last_decision = {
        **estimates,
        "strategy": strategy,
        "reason": reason,
    }

    return results, latency_ms, strategy
