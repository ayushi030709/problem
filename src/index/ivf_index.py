
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class IVFIndex:
    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        self.partitions = None
        self.vectors = None

    def build(self, vectors):
        """Cluster vectors and store product positions by partition."""
        self.vectors = np.asarray(vectors, dtype=np.float32)

        model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=2048,
            n_init=3,
        )

        labels = model.fit_predict(self.vectors)
        self.centroids = model.cluster_centers_

        self.partitions = [
            np.flatnonzero(labels == i)
            for i in range(self.n_clusters)
        ]

        return self

    def search(self, query, k=10, n_probe=5):
        """Search the nearest partitions and return top-k positions."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)

        if self.vectors is None:
            raise RuntimeError("Build the index before searching.")

        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query dimension does not match index.")

        n_probe = min(n_probe, self.n_clusters)

        centroid_scores = self.centroids @ query
        nearest = np.argpartition(
            centroid_scores, -n_probe
        )[-n_probe:]

        candidates = np.concatenate([
            self.partitions[i] for i in nearest
        ])

        if len(candidates) == 0:
            return np.array([], dtype=int)

        scores = self.vectors[candidates] @ query
        k = min(k, len(candidates))

        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return candidates[top]
