# Madurai Pincode Search
### Vector Search Benchmarking System

A Python-based benchmarking project for evaluating vector search strategies, metadata filtering, and adaptive query planning on a product dataset.

> **Project scope:** The current implementation uses Amazon product data to benchmark vector search. It does not yet provide a real Madurai postal pincode lookup service.

---

## 🎥 Project Demo Recordings

1. ▶️ **[Watch Demo Recording 1](https://drive.google.com/file/d/1cEw5Rd3o0KckiNY819jx6F3jilHamhnK/view?usp=sharing)**
2. ▶️ **[Watch Demo Recording 2](https://drive.google.com/file/d/1sWNAljhO5MjHaShuem6xzsB78y_UPqEm/view?usp=sharing)**

---


## 🚀 Key Features

- **Exact Vector Search** — Establishes a reference for retrieval quality.
- **IVF Approximate Search** — Evaluates search using an Inverted File index.
- **Metadata Filtering** — Applies filters to product search results.
- **Multiple Search Strategies** — Compares pre-filtering, post-filtering, and adaptive approaches.
- **Adaptive Query Planning** — Selects a search strategy based on query and filter characteristics.
- **Benchmark Evaluation** — Stores detailed measurements and summary results.
- **Mutation Testing** — Evaluates behavior under modified test conditions.
- **Reproducible Experiments** — Includes dataset manifests, benchmark outputs, and documentation.

## 🛠️ Technology Stack

- Python
- NumPy
- Pandas
- Pytest
- Vector similarity search
- IVF indexing
- Git and Git LFS

## 📁 Project Structure

```text
madurai-pincode-search-submission/
├── data/                 # Source and processed datasets
├── docs/                 # Architecture and results documentation
├── results/              # Benchmark and mutation-test outputs
├── scripts/              # Dataset, benchmark, and report scripts
├── src/
│   ├── benchmark/        # Benchmark components
│   ├── filters/          # Metadata filters and search strategies
│   ├── index/            # Exact search and IVF index
│   └── planner/           # Query planning
├── tests/                # Automated tests
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- Git LFS

### 1. Clone the repository

```bash
git clone https://github.com/ayushi030709/problem.git
cd problem
```

### 2. Download Git LFS files

Install Git LFS from [git-lfs.com](https://git-lfs.com/) if needed.

```bash
git lfs install
git lfs pull
```

### 3. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🧪 Running Tests

Execute the automated test suite from the repository root:

```bash
python -m pytest tests -v
```

The tests cover dataset handling, metadata filtering, indexing, and query-planning behavior.

## 📊 Running Benchmarks

Run the benchmark script:

```bash
python scripts/run_benchmark.py
```

Run mutation tests:

```bash
python scripts/run_mutation_test.py
```

Generate the results report:

```bash
python scripts/create_report.py
```

Refer to each script for supported options and configuration.

## 🔍 Search Strategies

| Strategy | Description |
|---|---|
| Exact Search | Exhaustively evaluates vectors to provide a retrieval reference. |
| IVF Approximate Search | Searches selected index partitions to reduce the search workload. |
| Pre-filtering | Filters candidates before vector search. |
| Post-filtering | Filters retrieved candidates after vector search. |
| Adaptive Strategy
