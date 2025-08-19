import numpy as np
import faiss


class TinyHNSW:
    def __init__(self, vector_dimensions, m=10, ef_construction=200, ef_search=50, seed=42):
        """
        Initialize TinyHNSW index.

        Args:
            vector_dimensions: Vector dimensionality
            m: Number of neighbors to consider for HNSW (default 10)
            ef_construction: Size of the dynamic candidate list during construction (default 200)
            ef_search: Size of the dynamic candidate list during search (default 50)
            seed: Random seed for reproducibility
        """
        self.vector_dimensions = vector_dimensions
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        np.random.seed(seed)

        # Create FAISS HNSW index
        self.index = faiss.IndexHNSWFlat(vector_dimensions, m)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # Storage for metadata
        self.id_to_metadata = {}
        self.next_id = 0

    def fit(self, items):
        """
        Add vectors to the index.

        Args:
            items: List of tuples (id, vector, metadata)
        """
        vectors = []
        for item_id, vector, item_metadata in items:
            # Convert vector to numpy array and ensure float32
            vector_array = np.array(vector, dtype=np.float32)
            if vector_array.ndim == 1:
                vector_array = vector_array.reshape(1, -1)
            vectors.append(vector_array.flatten())

            # Store metadata mapping
            self.id_to_metadata[self.next_id] = {
                'original_id': item_id,
                'metadata': item_metadata
            }
            self.next_id += 1

        # Convert to numpy array and add to index
        vectors_array = np.array(vectors, dtype=np.float32)
        self.index.add(x=vectors_array)

    def set_ef_search(self, ef_search):
        """Set the efSearch parameter for controlling search quality vs speed."""
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search

    def knn(self, target_vector, k=1, ef_search=None, filter_fn=None):
        """
        Search for k nearest neighbors.

        Args:
            target_vector: Query vector
            k: Number of nearest neighbors to return
            ef_search: Override efSearch parameter for this query
            filter_fn: Optional function to filter results based on metadata

        Returns:
            List of tuples (original_id, distance, metadata)
        """
        # Set temporary efSearch if provided
        original_ef_search = self.index.hnsw.efSearch
        if ef_search is not None:
            self.index.hnsw.efSearch = ef_search

        try:
            # Convert query vector to proper format
            query_array = np.array(target_vector, dtype=np.float32)
            if query_array.ndim == 1:
                query_array = query_array.reshape(1, -1)

            # Search for more results than needed if filtering
            search_k = k * 10 if filter_fn else k
            search_k = min(search_k, self.index.ntotal)  # Don't search for more than available

            # Perform search
            distances, indices = self.index.search(query_array, search_k)

            # Process results
            results = []
            for index in range(len(indices[0])):
                if indices[0][index] == -1:  # FAISS returns -1 for invalid results
                    break

                internal_id = indices[0][index]
                v_distance = float(distances[0][index])

                if internal_id in self.id_to_metadata:
                    original_id = self.id_to_metadata[internal_id]['original_id']
                    metadata = self.id_to_metadata[internal_id]['metadata']

                    # Apply filter if provided
                    if filter_fn is None or filter_fn(metadata):
                        results.append((original_id, v_distance, metadata))

                        # Stop when we have enough results
                        if len(results) >= k:
                            break

            return results[:k]

        finally:
            # Restore original efSearch
            self.index.hnsw.efSearch = original_ef_search

    def get_index_info(self):
        """Get information about the index."""
        return {
            'num_vectors': self.index.ntotal,
            'num_dimensions': self.vector_dimensions,
            'm': self.m,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'current_ef_search': self.index.hnsw.efSearch
        }


if __name__ == '__main__':
    # Example usage
    num_vectors = 1000
    num_dimensions = 128

    # Generate random vectors
    random_vectors = np.random.random((num_vectors, num_dimensions)).astype('float32')

    # Create TinyHNSW instance
    hnsw = TinyHNSW(num_dimensions, m=16, ef_construction=200, ef_search=50)

    # Prepare items with metadata
    items = []
    for i in range(num_vectors):
        metadata = {
            'id': i,
            'category': 'A' if i % 2 == 0 else 'B',
            'value': i * 10
        }
        items.append((f"doc_{i}", random_vectors[i], metadata))

    # Fit the index
    print("Fitting HNSW index...")
    hnsw.fit(items)

    # Create a query vector
    query_vector = np.random.random(num_dimensions).astype('float32')
    print(f"Query vector shape: {query_vector.shape}")

    # Test with different efSearch values
    print("\n=== HNSW Search Comparison ===")

    print("\nefSearch=10 (fast, lower recall):")
    results_10 = hnsw.knn(query_vector, k=5, ef_search=10)
    for orig_id, distance, metadata in results_10:
        print(f"  {orig_id}: distance={distance:.4f}, category={metadata['category']}")

    print("\nefSearch=100 (slower, higher recall):")
    results_100 = hnsw.knn(query_vector, k=5, ef_search=100)
    for orig_id, distance, metadata in results_100:
        print(f"  {orig_id}: distance={distance:.4f}, category={metadata['category']}")

    # Test filtered search
    print("\nFiltered search (category='A' only):")
    filter_fn = lambda meta: meta['category'] == 'A'
    filtered_results = hnsw.knn(query_vector, k=5, ef_search=100, filter_fn=filter_fn)
    for orig_id, distance, metadata in filtered_results:
        print(f"  {orig_id}: distance={distance:.4f}, category={metadata['category']}")

    # Print index info
    print(f"\nIndex info: {hnsw.get_index_info()}")
