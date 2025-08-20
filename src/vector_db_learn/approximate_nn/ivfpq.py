import faiss
import numpy as np


class TinyIVFPQ:
    """
    A simple wrapper around FAISS's IVFPQ index for approximate nearest neighbor search.
    This class allows you to create an index, add vectors, and perform k-NN searches
    with metadata support.

    Args:
        d (int): Dimension of the vectors.
        nlist (int): Number of coarse clusters (inverted lists).
        m (int): Number of sub-vectors for PQ (the vector will be split into m chunks).
        nbits (int): Number of bits per subquantizer (defines the codebook size: 2^nbits entries per sub-vector).
        nprobe (int): Number of clusters to search during a query (how many inverted lists to probe).
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, d, nlist=1024, m=16, nbits=8, nprobe=16, seed=42):
        self.nprobe = nprobe
        self.seed = seed

        # Build quantizer
        # - IVF needs a quantizer to assign each vector to a cluster centroid.
        # - Here: we’re using a flat L2 quantizer (standard choice).
        # - This doesn’t do compression itself; it’s just a way to bucket vectors into nlist clusters.
        self.quantizer = faiss.IndexFlatL2(d)

        # This is the real magic:
        # - First, IVF splits vectors into nlist clusters.
        # - Within each cluster, vectors are stored compressed using Product Quantization (PQ).
        # - m and nbits define how strong the compression is.
        #
        # Example:
        # If d=128, m=16, each subvector has size 128/16 = 8.
        # Each subvector is replaced by an index into a codebook of size 2^nbits.
        # With nbits=8, that’s 256 possible centroids per subvector.
        self.index = faiss.IndexIVFPQ(self.quantizer, d, nlist, m, nbits)
        # Since FAISS only stores numeric IDs internally, we’re maintaining a mapping:
        # id → metadata (e.g., original text, filenames, etc.).
        # next_id auto-increments as you add vectors.
        self.id_to_metadata = {}
        self.next_id = 0

    def train(self, vectors):
        """
        Train the IVF-PQ index.
        """
        if isinstance(vectors, (list, tuple)):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be a numpy array or list of arrays.")

        if vectors.ndim != 2:
            raise ValueError("Training vectors must be a 2D array of shape (n_samples, d).")

        self.index.train(vectors)

    def add(self, vectors, metadata=None):
        """
        Add vectors with optional metadata.
        """
        if isinstance(vectors, (list, tuple)):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be a numpy array or list of arrays.")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            raise ValueError("Vectors must be 1D or 2D arrays.")

        # Add to FAISS
        self.index.add(vectors)

        # Handle metadata
        n_new = vectors.shape[0]
        if metadata is None:
            metadata = [None] * n_new
        elif len(metadata) != n_new:
            raise ValueError("Metadata length must match number of vectors.")

        for meta in metadata:
            self.id_to_metadata[self.next_id] = meta
            self.next_id += 1

    def knn(self, query_vector, k=5):
        """
        Perform k-NN search on the index.

        Args:
            query_vector (np.ndarray): The query vector (1D array).
            k (int): Number of nearest neighbors to return.

        Returns:
            List of tuples (id, distance, metadata) for the k nearest neighbors.
        """
        if not isinstance(query_vector, np.ndarray):
            raise ValueError("Query vector must be a numpy array.")
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array.")

        # Ensure the index is trained
        if not self.index.is_trained:
            raise RuntimeError("Index is not trained. Call train() before searching.")

        # Reshape query vector to 2D for FAISS
        query_vector_2d = query_vector.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vector_2d, k)

        # Collect scores with metadata
        scores = []
        for i in range(k):
            idx = indices[0][i]
            if idx == -1:  # FAISS returns -1 for invalid scores
                continue
            distance = distances[0][i]
            metadata = self.id_to_metadata.get(idx, None)
            scores.append((idx, distance, metadata))
        return scores


if __name__ == '__main__':
    # Example usage
    d = 128  # Dimension of vectors
    nlist = 10  # Number of clusters
    m = 8  # Number of sub-vectors for PQ
    nbits = 8  # Bits per sub-vector
    nprobe = 5  # Number of clusters to search

    ivf_pq_index = TinyIVFPQ(d, nlist, m, nbits, nprobe)

    # Generate some random training data
    np.random.seed(42)
    training_vectors = np.random.rand(1000, d).astype(np.float32)

    # Train the index
    ivf_pq_index.train(training_vectors)

    # Add some vectors with metadata
    for i in range(100):
        vector = np.random.rand(d).astype(np.float32)
        metadata = {"id": i, "info": f"Vector {i}"}
        ivf_pq_index.add(vector.reshape(1, -1), metadata=[metadata])  # Fix: wrap metadata in a list

    # Perform a k-NN search
    query_vector = np.random.rand(d).astype(np.float32)
    results = ivf_pq_index.knn(query_vector, k=5)

    print("Search results:")
    for res in results:
        print(res)