import math


class TinyVectorStore:
    def __init__(self):
        self.vectors = {}  # key -> vector

    def upsert(self, key, vector):
        """Insert or update a vector."""
        self.vectors[key] = vector

    @staticmethod
    def cosine_similarity(u, v):
        # Calculate the dot product first
        dot_product = sum(u_i * v_i for u_i, v_i in zip(u, v))
        # Calculate the magnitudes of the vectors
        magnitude_u = math.sqrt(sum(u_i * u_i for u_i in u))
        magnitude_v = math.sqrt(sum(v_i * v_i for v_i in v))
        # Handle the case where one of the vectors is zero
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0
        # Return the cosine similarity
        return dot_product / (magnitude_u * magnitude_v)

    def knn(self, target_vector, k=1):
        scores = []
        for key, vector in self.vectors.items():
            score = self.cosine_similarity(target_vector, vector)
            scores.append((key, score))
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        # Return the top k results
        return scores[:k]

if __name__ == '__main__':
    store = TinyVectorStore()
    store.upsert('doc1', [1, 2, -3])
    store.upsert('doc2', [4, -5, 6])
    store.upsert('doc3', [7, -8, 9])

    query_vector = [1, 0, -1]
    results = store.knn(query_vector, k=2)
    print(results)  # Should print the top 2 most similar vectors to the query vector