# Madurai Pincode Search — Results Report

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

| target_selectivity   | strategy    |   actual_selectivity |   eligible_products |   mean_recall_at_10 |   p50_latency_ms |   p95_latency_ms |   mean_returned |
|:---------------------|:------------|---------------------:|--------------------:|--------------------:|-----------------:|-----------------:|----------------:|
| 100%                 | post-filter |               1.0000 |          30000.0000 |              0.9580 |           1.6081 |           2.2445 |         10.0000 |
| 100%                 | pre-filter  |               1.0000 |          30000.0000 |              1.0000 |          23.1588 |          30.1642 |         10.0000 |
| 100%                 | adaptive    |               1.0000 |          30000.0000 |              0.9580 |           1.7273 |           2.7311 |         10.0000 |
| 10%                  | post-filter |               0.1005 |           3014.0000 |              0.3640 |           1.5471 |           2.0753 |          3.9600 |
| 10%                  | pre-filter  |               0.1005 |           3014.0000 |              1.0000 |           2.1150 |           2.8517 |         10.0000 |
| 10%                  | adaptive    |               0.1005 |           3014.0000 |              1.0000 |           2.5011 |           3.0498 |         10.0000 |
| 1%                   | post-filter |               0.0100 |            300.0000 |              0.0880 |           1.5463 |           2.1506 |          0.9400 |
| 1%                   | pre-filter  |               0.0100 |            300.0000 |              1.0000 |           0.2856 |           0.3272 |         10.0000 |
| 1%                   | adaptive    |               0.0100 |            300.0000 |              1.0000 |           0.5570 |           0.5983 |         10.0000 |
| 0.1%                 | post-filter |               0.0010 |             30.0000 |              0.0000 |           1.4442 |           2.1448 |          0.0000 |
| 0.1%                 | pre-filter  |               0.0010 |             30.0000 |              1.0000 |           0.2368 |           0.2980 |         10.0000 |
| 0.1%                 | adaptive    |               0.0010 |             30.0000 |              1.0000 |           0.5011 |           0.5790 |         10.0000 |
| 0.01%                | post-filter |               0.0001 |              3.0000 |              0.0000 |           1.6043 |           2.1989 |          0.0000 |
| 0.01%                | pre-filter  |               0.0001 |              3.0000 |              1.0000 |           0.2346 |           0.2615 |          3.0000 |
| 0.01%                | adaptive    |               0.0001 |              3.0000 |              1.0000 |           0.5021 |           0.5535 |          3.0000 |

## 3. Mutation experiment

The mutation experiment deletes 10% of the products in the busiest zone
and adds 5% of the original dataset as synthetic products with perturbed
embeddings and shifted metadata. The original dataset files are preserved.

- Busiest zone: ZONE-06
- Deleted from busiest zone: 248
- Added products: 1500
- Total mutated dataset size: 31252

### Mutation results

| stage   | strategy    |   queries |   mean_recall_at_10 |   p50_latency_ms |   p95_latency_ms |   mean_returned |
|:--------|:------------|----------:|--------------------:|-----------------:|-----------------:|----------------:|
| after   | adaptive    |        50 |              1.0000 |           4.4379 |           4.9773 |         10.0000 |
| after   | post-filter |        50 |              0.2800 |           2.8010 |           3.9138 |          3.1400 |
| after   | pre-filter  |        50 |              1.0000 |           2.9068 |           3.6881 |         10.0000 |
| before  | adaptive    |        50 |              1.0000 |           4.5734 |           6.0531 |         10.0000 |
| before  | post-filter |        50 |              0.3100 |           2.7333 |           3.6259 |          3.7000 |
| before  | pre-filter  |        50 |              1.0000 |           3.1209 |           4.0007 |         10.0000 |

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
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe scripts\run_benchmark.py
.\.venv\Scripts\python.exe scripts\run_mutation_test.py
```

## 7. Output files

- `results/benchmark_detail.csv`
- `results/benchmark_summary.csv`
- `results/benchmark_manifest.json`
- `results/mutation_detail.csv`
- `results/mutation_summary.csv`
- `results/mutation_manifest.json`
- `docs/architecture.md`
