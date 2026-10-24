
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 42
N = 30_000

# 1. Load category information
categories = pd.read_csv(DATA / "amazon_categories.csv")

# 2. Read products in chunks and keep valid records
parts = []
for chunk in pd.read_csv(
    DATA / "amazon_products.csv",
    usecols=["asin", "title", "price", "category_id"],
    chunksize=50_000,
):
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
    chunk["category_id"] = pd.to_numeric(
        chunk["category_id"], errors="coerce"
    )
    chunk = chunk.dropna(
        subset=["asin", "title", "price", "category_id"]
    )
    chunk = chunk[
        chunk["title"].str.strip().ne("") & (chunk["price"] > 0)
    ]
    parts.append(chunk)

products = pd.concat(parts, ignore_index=True)
products = products.drop_duplicates("asin")

# 3. Keep the 20 most common categories
products = products.merge(
    categories, left_on="category_id", right_on="id", how="inner"
)
top = products["category_name"].value_counts().head(20).index
products = products[products["category_name"].isin(top)]

products = products.sample(n=N, random_state=SEED).reset_index(drop=True)
products = products.rename(
    columns={"asin": "product_id", "category_name": "category"}
)
products["category_id"] = products["category_id"].astype(int)

# 4. Generate 30 benchmark zones, correlated with category
rng = np.random.default_rng(SEED)
category_codes = products["category"].astype("category").cat.codes.to_numpy()
preferred = category_codes % 30
zones = np.where(
    rng.random(N) < 0.9,
    preferred,
    rng.integers(0, 30, N),
)
products["zone"] = [f"ZONE-{z + 1:02d}" for z in zones]

products = products[
    ["product_id", "title", "category", "category_id", "price", "zone"]
]

# 5. Generate semantic vectors from product titles
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(
    products["title"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)

# 6. Save the dataset, vectors and basic manifest
products.to_csv(DATA / "products_clean.csv", index=False)
np.save(DATA / "product_embeddings.npy", vectors)

manifest = {
    "products": len(products),
    "categories": int(products["category"].nunique()),
    "zones": int(products["zone"].nunique()),
    "embedding_dimensions": int(vectors.shape[1]),
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "seed": SEED,
    "zones_are_synthetic": True,
}
(DATA / "dataset_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("Dataset created:", len(products), "products")
print("Categories:", products["category"].nunique())
print("Zones:", products["zone"].nunique())
print("Vector shape:", vectors.shape)
