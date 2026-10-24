
from pathlib import Path
import json
import time
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.ivf_index import IVFIndex
from src.index.exact_search import exact_search
from src.filters.search_strategies import (
    post_filter_search,
    pre_filter_search,
    adaptive_search,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

K = 10
SEED = 42
N_QUERIES = 50


def recall_at_k(predicted, ground_truth):
    """Compare results with ground truth, handling fewer than k eligible items."""
    truth = set(ground_truth.tolist())
    if not truth:
        return 1.0 if len(predicted) == 0 else 0.0
    return len(set(predicted.tolist()) & truth) / len(truth)


def percentile_latency(values, percentile):
    return float(np.percentile(values, percentile))


def main():
    RESULTS.mkdir(exist_ok=True)

    products = pd.read_csv(DATA / "products_clean.csv")
    vectors = np.load(DATA / "product_embeddings.npy").astype(np.float32)

    if len(products) != len(vectors):
        raise ValueError("Product count does not match embedding count.")

    print(f"Loaded {len(products):,} products")
    print(f"Embedding dimensions: {vectors.shape[1]}")

    # Build the IVF index once, outside the timed search measurements.
    n_clusters = min(100, len(vectors))
    index = IVFIndex(n_clusters=n_clusters).build(vectors)
    print(f"IVF index built with {n_clusters} partitions")

    # Generate queries from real product embeddings, using a fixed seed.
    rng = np.random.default_rng(SEED)
    query_positions = rng.choice(
        len(vectors), size=N_QUERIES, replace=False
    )

    # Build filters whose observed selectivity is near the requested level.
    # We use a price threshold because it can produce several selectivity levels.
    price_values = products["price"].to_numpy(dtype=float)
    target_levels = [
        ("100%", 1.0),
        ("10%", 0.10),
        ("1%", 0.01),
        ("0.1%", 0.001),
        ("0.01%", 0.0001),
    ]

    filter_cases = []
    for label, target in target_levels:
        if target == 1.0:
            filters = None
        else:
            threshold = float(np.quantile(price_values, 1.0 - target))
            filters = {"min_price": threshold}

        mask = (
            np.ones(len(products), dtype=bool)
            if not filters
            else products["price"].ge(filters["min_price"]).to_numpy()
        )
        eligible_count = int(mask.sum())
        actual_selectivity = eligible_count / len(products)

        filter_cases.append(
            (label, filters, mask, eligible_count, actual_selectivity)
        )

    rows = []

    for label, filters, mask, eligible_count, actual_selectivity in filter_cases:
        print(
            f"\nTarget {label}: {eligible_count:,} eligible products "
            f"({actual_selectivity:.4%} actual)"
        )

        for query_number, position in enumerate(query_positions, start=1):
            query = vectors[position]

            # Exact brute-force ground truth. This timing is not included
            # in the strategy latency measurements.
            truth = exact_search(
                products, vectors, query, filters=filters, k=K
            )

            strategy_runs = {}

            start = time.perf_counter()
            results, _ = post_filter_search(
                products, vectors, query, index, filters=filters, k=K
            )
            strategy_runs["post-filter"] = (
                results, (time.perf_counter() - start) * 1000
            )

            start = time.perf_counter()
            results, _ = pre_filter_search(
                products, vectors, query, filters=filters, k=K
            )
            strategy_runs["pre-filter"] = (
                results, (time.perf_counter() - start) * 1000
            )

            start = time.perf_counter()
            results, _, chosen = adaptive_search(
                products, vectors, query, index, filters=filters, k=K
            )
            strategy_runs["adaptive"] = (
                results, (time.perf_counter() - start) * 1000
            )

            for strategy, (predicted, latency_ms) in strategy_runs.items():
                rows.append({
                    "target_selectivity": label,
                    "actual_selectivity": actual_selectivity,
                    "eligible_products": eligible_count,
                    "query_number": query_number,
                    "query_product_id": products.iloc[position]["product_id"],
                    "strategy": strategy,
                    "planner_choice": (
                        chosen if strategy == "adaptive" else ""
                    ),
                    "ground_truth_count": len(truth),
                    "returned_count": len(predicted),
                    "recall_at_10": recall_at_k(predicted, truth),
                    "latency_ms": latency_ms,
                })

    detail = pd.DataFrame(rows)
    detail_path = RESULTS / "benchmark_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["target_selectivity", "strategy"], sort=False)
        .agg(
            actual_selectivity=("actual_selectivity", "mean"),
            eligible_products=("eligible_products", "mean"),
            mean_recall_at_10=("recall_at_10", "mean"),
            p50_latency_ms=("latency_ms", lambda x: percentile_latency(x, 50)),
            p95_latency_ms=("latency_ms", lambda x: percentile_latency(x, 95)),
            mean_returned=("returned_count", "mean"),
        )
        .reset_index()
    )

    summary_path = RESULTS / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "seed": SEED,
        "products": len(products),
        "embedding_dimensions": int(vectors.shape[1]),
        "queries_per_selectivity": N_QUERIES,
        "k": K,
        "ivf_partitions": n_clusters,
        "ground_truth": "exact filtered brute-force cosine/dot-product ranking",
        "notes": [
            "Vectors and query embeddings are normalized by the dataset generator.",
            "Selectivity uses observed price thresholds; actual values may differ from targets.",
            "Latency includes strategy execution, but excludes IVF index build and ground-truth computation.",
        ],
    }
    (RESULTS / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nBenchmark summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved detailed results: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved manifest: {RESULTS / 'benchmark_manifest.json'}")


if __name__ == "__main__":
    main()
