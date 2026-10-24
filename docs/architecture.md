
# Madurai Pincode Search — Architecture

## System overview

```text
Product CSV + Category CSV
            |
            v
     Dataset Generator
            |
            +----------------------+
            |                      |
            v                      v
    Clean Product Data       Text Embeddings
    (CSV metadata)           (NumPy array)
            |                      |
            +-----------+----------+
                        |
                        v
                  IVF Index
             (MiniBatchKMeans)
                        |
User Query + Metadata Filters
                        |
                        v
                 Search Planner
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
       Post-filter  Pre-filter  Adaptive
            |           |           |
            +-----------+-----------+
                        |
                        v
                 Ranked Results
                        |
                        v
             Benchmark + Evaluation
             Recall@10, p50, p95
```

## Main components

- `scripts/generate_dataset.py`: cleans the source product data and generates normalized text embeddings.
- `src/index/ivf_index.py`: builds an IVF-style index using MiniBatchKMeans partitions.
- `src/index/exact_search.py`: computes exact top-k results for ground truth.
- `src/filters/metadata_filters.py`: applies category, zone and price filters.
- `src/filters/search_strategies.py`: implements post-filter, pre-filter and adaptive search.
- `scripts/run_benchmark.py`: compares search strategies at five selectivity levels.
- `scripts/run_mutation_test.py`: tests search after deletion and distribution shift.
- `tests/`: verifies index, filters, strategies and planner behavior.

## Reproducibility

Run these commands from the project root using the virtual environment's Python executable:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe scripts\run_benchmark.py
.\.venv\Scripts\python.exe scripts\run_mutation_test.py
```

The dataset generator uses seed 42. Benchmark and mutation scripts also use fixed random seeds.

## Evaluation metrics

- Recall@10: overlap between strategy results and exact ground-truth top-k results.
- p50 latency: median measured search latency.
- p95 latency: 95th-percentile measured search latency.

## Known limitations

- The IVF implementation uses a simple centroid-partition search and probes a limited number of partitions.
- The adaptive planner uses an initial estimated-work model and should not be described as globally optimal.
- Zones are synthetic metadata rather than real geographic pincodes.
- Mutation uses synthetic embedding perturbations to model distribution shift.
