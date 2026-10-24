
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

benchmark = pd.read_csv(RESULTS / "benchmark_summary.csv")
mutation = pd.read_csv(RESULTS / "mutation_summary.csv")
manifest = pd.read_json(RESULTS / "mutation_manifest.json", typ="series")

def markdown_table(frame):
    return frame.to_markdown(index=False, floatfmt=".4f")

report = f"""# Madurai Pincode Search — Results Report

## 1. Project overview

This project compares three metadata-filtered vector-search strategies
using 30,000 product records and 384-dimensional text embeddings.

- **Post-filtering:** Retrieve approximate IVF neighbors, then apply metadata filters.
- **Pre-filtering:** Apply metadata filters, then rank eligible vectors exactly.
- **Adaptive:** Choose a strategy using estimated search work and a selective-filter quality safeguard.

The IVF index uses MiniBatchKMeans with 100 partitions. Embeddings are generated
with `sentence-transformers/all-MiniLM-L6-v2`.

## 2. Benchmark methodology

The benchmark uses 50 query embeddings per target selectivity level.
Ground truth is obtained by exact filtered search. Recall@10 is the overlap
between a strategy's results and the exact top-k results, divided by the
number of ground-truth results available.

Latency is measured in milliseconds. p50 is the median; p95 is the 95th percentile.
Index construction and ground-truth computation are excluded from strategy latency.

### Benchmark results

{markdown_table(benchmark)}

## 3. Mutation experiment

The mutation experiment deletes 10% of the products in the busiest zone
and adds 5% of the original dataset as synthetic products with perturbed
embeddings and shifted metadata. The original dataset files are preserved.

- Busiest zone: {manifest.get('busiest_zone', 'not recorded')}
- Deleted from busiest zone: {manifest.get('deleted_from_busiest_zone', 'not recorded')}
- Added products: {manifest.get('added_products', 'not recorded')}
- Total mutated dataset size: {manifest.get('mutated_products_total', 'not recorded')}

### Mutation results

{markdown_table(mutation)}

In this experiment, the adaptive strategy retained 100% recall before and
after mutation. Post-filtering had lower recall because the limited IVF
candidate set did not contain all exact eligible neighbors. These results
apply to the query and filter conditions used by the mutation script.

## 4. Planner-failure experiment

The deterministic planner test used 1,000 normalized vectors and a price
filter with 10% selectivity.

- Old fixed-threshold choice: post-filter.
- Old post-filter recall@10: 60%.
- New adaptive choice: pre-filter.
- New adaptive recall@10: 100%.

The fixed-threshold planner failed because it considered selectivity alone
and did not account for whether approximate retrieval would find enough
eligible nearest neighbors. The new planner chose exact pre-filtering for
this selective case.

## 5. Conclusions

1. Post-filtering is fast when the filter is broad, but it can return too few
   eligible results when filters are selective.
2. Exact pre-filtering gives the exact top results among eligible products.
   Its cost depends on the number of eligible vectors.
3. The adaptive planner improved recall in the tested selective cases.
4. Cost estimates are approximate and have not been calibrated into a
   universally optimal latency predictor.
5. Zones are synthetic metadata. The mutation experiment uses controlled
   embedding perturbations rather than embeddings generated from newly
   collected product text.

## 6. Reproducibility

Run these commands from the project root:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -v
.\\.venv\\Scripts\\python.exe scripts\\run_benchmark.py
.\\.venv\\Scripts\\python.exe scripts\\run_mutation_test.py
```

## 7. Output files

- `results/benchmark_detail.csv`
- `results/benchmark_summary.csv`
- `results/benchmark_manifest.json`
- `results/mutation_detail.csv`
- `results/mutation_summary.csv`
- `results/mutation_manifest.json`
- `docs/architecture.md`
"""

path = DOCS / "results_report.md"
path.write_text(report, encoding="utf-8")
print(f"Report created: {path}")
print("\nBenchmark rows:", len(benchmark))
print("Mutation rows:", len(mutation))
