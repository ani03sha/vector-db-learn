import math
import random


def l2(u, v):
    """Calculate the L2 distance between two vectors."""
    return math.sqrt(sum((u_i - v_i) ** 2 for u_i, v_i in zip(u, v)))


def normalize(vector):
    """Normalize a vector."""
    magnitude = math.sqrt(sum(x ** 2 for x in vector))
    if magnitude == 0:
        return [0] * len(vector)
    return [x / magnitude for x in vector]


class TinyIVF:
    def __init__(self, nlist=10, iters=10, seed=42):
        self.nlist = nlist
        self.iters = iters
        self.rng = random.Random(seed)
        self.centroids = []  # List of centroids for each cluster - list[list[float]]
        self.clusters = {}  # Dictionary to map each vector to its cluster - centroid_index -> list[(id, vector, metadata)]

    def _init_centroids(self, vectors):
        # KMeans++ initialization - pick one random vector as the first centroid
        self.centroids = [vectors[self.rng.randrange(len(vectors))][:]]
        while len(self.centroids) < self.nlist:
            distances = []
            for vector in vectors:
                d = min(l2(vector, centroid) for centroid in self.centroids)
                distances.append(d ** 2)  # Square the distance for better separation
            total_distance = sum(distances) or 1.0
            r = self.rng.random() * total_distance
            cumulative = 0.0
            for vector, distance in zip(vectors, distances):
                cumulative += distance
                if cumulative >= r:
                    self.centroids.append(vector[:])
                    break

    def _assign_cluster(self, vectors):
        # Return index of the closest centroid for each vector
        best_centroid, best_distance = -1, float('inf')
        for i, vector in enumerate(self.centroids):
            distance = l2(vectors, vector)
            if distance < best_distance:
                best_centroid, best_distance = i, distance
        return best_centroid

    def fit(self, items):
        """
        :param items: list of tuples (id, vector, metadata)
        """
        vectors = [normalize(vector) for _, vector, _ in items]
        if len(vectors) < self.nlist:
            self.nlist = max(1, len(vectors))
        self._init_centroids(vectors)

        # Lloyd's iteration
        for _ in range(self.iters):
            buckets = [[] for _ in range(self.nlist)]
            for vector in vectors:
                buckets[self._assign_cluster(vector)].append(vector)
            # Recompute centroids
            new_centroids = []
            for bucket in buckets:
                if not bucket:
                    # Re-seed a centroid if the bucket is empty
                    new_centroids.append(vectors[self.rng.randrange(len(vectors))][:])
                else:
                    distance = len(bucket[0])
                    mean = [0.0] * distance
                    for v in bucket:
                        for j in range(distance):
                            mean[j] += v[j]
                    mean = [x / len(bucket) for x in mean]
                    new_centroids.append(mean)
            self.centroids = new_centroids

        # Final assignment of vectors to clusters
        self.clusters = {i: [] for i in range(self.nlist)}
        for (id_, vector, metadata) in items:
            v = normalize(vector)
            centroid_index = self._assign_cluster(v)
            self.clusters[centroid_index].append((id_, v, metadata))

    def nearest_neighbors(self, q, nprobe):
        """Return indices of the nprobe closest centroids to q (q should be normalized)."""
        q = normalize(q)
        distances = []
        for i, centroid in enumerate(self.centroids):
            distance = l2(q, centroid)
            distances.append((i, distance))
        # Sort by distance and take the first nprobe indices
        distances.sort(key=lambda x: x[1])
        return [index for index, _ in distances[:nprobe]]

    @staticmethod
    def cosine(u, v):
        # Dot product on unit vectors
        return sum(u_i * v_i for u_i, v_i in zip(u, v))


    def knn(self, query_vector, k=1, nprobe=None, filter_fn=None):
        if nprobe is None:
            nprobe = self.nlist
        query_vector = normalize(query_vector)
        probe_indexes = self.nearest_neighbors(query_vector, nprobe)
        scores = []
        for probe_index in probe_indexes:
            for (id_, vector, metadata) in self.clusters[probe_index]:
                if filter_fn is None or filter_fn(metadata):
                    score = self.cosine(query_vector, vector)
                    scores.append((id_, score, metadata))
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        # Return the top k results
        return scores[:k]


