
from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index.ivf_index import IVFIndex
from src.index.exact_search import exact_search
from src.filters.search_strategies import (
    post_filter_search,
    pre_filter_search,
    adaptive_search,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42
K = 10
N_QUERIES = 50


def recall_at_k(predicted, truth):
    truth_set = set(truth.tolist())
    if not truth_set:
        return 1.0 if len(predicted) == 0 else 0.0
    return len(set(predicted.tolist()) & truth_set) / len(truth_set)


def measure(products, vectors, index, query_positions, stage):
    rows = []

    for query_number, position in enumerate(query_positions, start=1):
        query = vectors[position]
        filters = {"zone": "ZONE-06"}

        truth = exact_search(
            products, vectors, query, filters=filters, k=K
        )

        runs = {}

        start = time.perf_counter()
        result, _ = post_filter_search(
            products, vectors, query, index, filters=filters, k=K
        )
        runs["post-filter"] = (
            result, (time.perf_counter() - start) * 1000
        )

        start = time.perf_counter()
        result, _ = pre_filter_search(
            products, vectors, query, filters=filters, k=K
        )
        runs["pre-filter"] = (
            result, (time.perf_counter() - start) * 1000
        )

        start = time.perf_counter()
        result, _, choice = adaptive_search(
            products, vectors, query, index, filters=filters, k=K
        )
        runs["adaptive"] = (
            result, (time.perf_counter() - start) * 1000
        )

        for strategy, (predicted, latency) in runs.items():
            rows.append({
                "stage": stage,
                "query_number": query_number,
                "strategy": strategy,
                "planner_choice": choice if strategy == "adaptive" else "",
                "recall_at_10": recall_at_k(predicted, truth),
                "latency_ms": latency,
                "returned_count": len(predicted),
                "ground_truth_count": len(truth),
            })

    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    products = pd.read_csv(DATA / "products_clean.csv")
    vectors = np.load(DATA / "product_embeddings.npy").astype(np.float32)

    if len(products) != len(vectors):
        raise ValueError("Product and embedding counts do not match.")

    original_products = products.reset_index(drop=True).copy()
    original_vectors = vectors.copy()

    # Identify the busiest zone and delete 10% of its products.
    zone_counts = products["zone"].value_counts()
    busiest_zone = zone_counts.index[0]
    zone_positions = np.flatnonzero(
        products["zone"].to_numpy() == busiest_zone
    )

    delete_count = max(1, int(round(len(zone_positions) * 0.10)))
    delete_positions = rng.choice(
        zone_positions, size=delete_count, replace=False
    )

    keep_mask = np.ones(len(products), dtype=bool)
    keep_mask[delete_positions] = False

    products = products.iloc[np.flatnonzero(keep_mask)].reset_index(drop=True)
    vectors = vectors[keep_mask]

    # Add 5% new products with shifted vector distribution.
    add_count = int(round(len(original_products) * 0.05))
    source_positions = rng.choice(
        len(original_products), size=add_count, replace=True
    )

    new_vectors = original_vectors[source_positions].copy()
    shift = rng.normal(
        loc=0.0, scale=0.12, size=new_vectors.shape
    ).astype(np.float32)
    new_vectors = new_vectors + shift
    new_vectors /= np.maximum(
        np.linalg.norm(new_vectors, axis=1, keepdims=True), 1e-12
    )

    new_products = original_products.iloc[source_positions].copy()
    new_products = new_products.reset_index(drop=True)
    new_products["product_id"] = [
        f"MUTATED-{i + 1:05d}" for i in range(add_count)
    ]
    new_products["price"] = (
        new_products["price"].to_numpy(dtype=float) * 1.25
    )
    new_products["zone"] = [
        f"NEW-ZONE-{(i % 5) + 1:02d}" for i in range(add_count)
    ]
    new_products["title"] = (
        "Shifted distribution product " + new_products["title"].astype(str)
    )

    products = pd.concat(
        [products, new_products], ignore_index=True
    )
    vectors = np.vstack([vectors, new_vectors]).astype(np.float32)

    if len(products) != len(vectors):
        raise ValueError("Mutation produced misaligned products and vectors.")

    # Build the index separately for the mutated dataset.
    index = IVFIndex(n_clusters=100).build(vectors)

    # Sample queries from the original dataset that remain in the mutated set.
    surviving_ids = set(products["product_id"].astype(str))
    eligible_original = np.flatnonzero(
        original_products["product_id"].astype(str).isin(surviving_ids).to_numpy()
    )
    query_positions_original = rng.choice(
        eligible_original,
        size=min(N_QUERIES, len(eligible_original)),
        replace=False,
    )

    # Map the sampled original product IDs to their positions in mutated data.
    query_ids = original_products.iloc[query_positions_original]["product_id"].astype(str)
    id_to_position = {
        str(product_id): i
        for i, product_id in enumerate(products["product_id"])
    }
    query_positions_mutated = np.array(
        [id_to_position[product_id] for product_id in query_ids],
        dtype=int,
    )

    # Measure both datasets with their own indexes and queries.
    original_index = IVFIndex(n_clusters=100).build(original_vectors)
    original_rows = measure(
        original_products,
        original_vectors,
        original_index,
        query_positions_original,
        "before",
    )
    mutated_rows = measure(
        products,
        vectors,
        index,
        query_positions_mutated,
        "after",
    )

    detail = pd.DataFrame(original_rows + mutated_rows)
    detail.to_csv(RESULTS / "mutation_detail.csv", index=False)

    summary = (
        detail.groupby(["stage", "strategy"])
        .agg(
            queries=("query_number", "count"),
            mean_recall_at_10=("recall_at_10", "mean"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=(
                "latency_ms",
                lambda x: float(np.percentile(x, 95)),
            ),
            mean_returned=("returned_count", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "mutation_summary.csv", index=False)

    manifest = {
        "seed": SEED,
        "original_products": len(original_products),
        "busiest_zone": str(busiest_zone),
        "busiest_zone_original_count": int(len(zone_positions)),
        "deleted_from_busiest_zone": int(delete_count),
        "added_products": int(add_count),
        "mutated_products_total": int(len(products)),
        "embedding_dimensions": int(vectors.shape[1]),
        "mutation": {
            "delete_fraction_of_busiest_zone": 0.10,
            "add_fraction_of_original_dataset": 0.05,
            "embedding_shift_noise_std": 0.12,
            "new_product_price_multiplier": 1.25,
            "new_zones": 5,
        },
        "original_data_modified": False,
    }
    (RESULTS / "mutation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Busiest zone: {busiest_zone}")
    print(f"Deleted: {delete_count:,} products from that zone")
    print(f"Added: {add_count:,} shifted products")
    print(f"Mutated dataset size: {len(products):,}")
    print("\nMutation summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved results/mutation_detail.csv")
    print("Saved results/mutation_summary.csv")
    print("Saved results/mutation_manifest.json")


if __name__ == "__main__":
    main()
