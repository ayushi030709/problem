
import numpy as np

from src.filters.metadata_filters import apply_filters


def exact_search(products, vectors, query, filters=None, k=10):
    """Return exact top-k products satisfying all metadata filters."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float32)

    if len(products) != len(vectors):
        raise ValueError("Product and vector counts must match.")

    if query.shape[0] != vectors.shape[1]:
        raise ValueError("Query dimension does not match vectors.")

    mask = apply_filters(products, filters).to_numpy()
    eligible = np.flatnonzero(mask)

    if len(eligible) == 0:
        return np.array([], dtype=int)

    scores = vectors[eligible] @ query
    k = min(k, len(eligible))

    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]

    return eligible[top]
