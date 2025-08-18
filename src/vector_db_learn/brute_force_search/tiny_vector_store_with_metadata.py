import math


class TinyVectorStoreWithMetadata:
    def __init__(self):
        self.vectors = {}  # Changed from [] to {} to store key-value pairs

    def upsert(self, key, vector, metadata):
        self.vectors[key] = {"vector": vector, "metadata": metadata}

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

    def knn(self, query, k=1, filter_fn=None):
        # Step 1: Apply metadata filtering if needed
        filtered_vectors = {}
        for key, item in self.vectors.items():
            if filter_fn is None or filter_fn(item["metadata"]):
                filtered_vectors[key] = item

        # Step 2: Calculate similarity scores for filtered vectors
        candidates = []
        for key, item in filtered_vectors.items():
            similarity_score = self.cosine_similarity(query, item["vector"])
            candidates.append((key, similarity_score))

        # Step 3: Sort by similarity score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Return top k results
        return candidates[:k]