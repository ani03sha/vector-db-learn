from sentence_transformers import SentenceTransformer

# Handle both relative and absolute imports
try:
    from ..vector_similarity.tiny_vector_store import TinyVectorStore
    from ..brute_force_search.tiny_vector_store_with_metadata import TinyVectorStoreWithMetadata
    from ..approximate_nn.ivf import TinyIVF
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from vector_similarity.tiny_vector_store import TinyVectorStore
    from brute_force_search.tiny_vector_store_with_metadata import TinyVectorStoreWithMetadata
    from approximate_nn.ivf import TinyIVF


class VectorStoreRunner:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the runner with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.sample_data = [
            "Die Hard is an action movie about a cop fighting terrorists in a skyscraper.",
            "Lethal Weapon is a buddy-cop action film with lots of explosions.",
            "Frozen is a Disney movie about two sisters and magical ice powers."
        ]

    def encode_texts(self, texts=None):
        """Convert text data to normalized embeddings."""
        if texts is None:
            texts = self.sample_data

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return texts, embeddings


    @staticmethod
    def display_embeddings(texts, embeddings):
        """Display the first 5 dimensions of each embedding."""
        for text, vector in zip(texts, embeddings):
            print(text, "\nVector (first 5 dims):", vector[:5], "\n")

    def demo_basic_vector_store(self, texts, embeddings, query="explosions and criminals", k=2):
        """Demonstrate basic vector store functionality."""
        print("=== Basic Vector Store Demo ===")

        # Create and populate store
        store = TinyVectorStore()
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            store.upsert(f"doc{i+1}", vector.tolist())

        # Search
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        matches = store.knn(query_vector, k=k)

        print(f"Query: '{query}'")
        print("Top matches:", matches)
        return matches

    def demo_metadata_vector_store(self, texts, embeddings, query="ice age", k=2):
        """Demonstrate vector store with metadata functionality."""
        print("\n=== Vector Store with Metadata Demo ===")

        # Create and populate store with metadata
        store = TinyVectorStoreWithMetadata()
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = {
                "title": f"Document {i+1}",
                "category": "action" if "action" in text.lower() else "other"
            }
            store.upsert(f"doc{i+1}", vector.tolist(), metadata)

        # Search
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        matches = store.knn(query_vector, k=k)

        print(f"Query: '{query}'")
        print("Top matches with metadata:", matches)
        return matches

    def demo_filtered_search(self, texts, embeddings, query="action movie", category_filter="action", k=2):
        """Demonstrate filtered search using metadata."""
        print("\n=== Filtered Search Demo ===")

        # Create and populate store with metadata
        store = TinyVectorStoreWithMetadata()
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = {
                "title": f"Document {i+1}",
                "category": "action" if "action" in text.lower() else "other"
            }
            store.upsert(f"doc{i+1}", vector.tolist(), metadata)

        # Search with filter
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        filter_fn = lambda meta: meta.get("category") == category_filter
        matches = store.knn(query_vector, k=k, filter_fn=filter_fn)

        print(f"Query: '{query}' (filtered by category: {category_filter})")
        print("Filtered matches:", matches)
        return matches

    def demo_ivf_vector_store(self, texts, embeddings, query="explosions and criminals", k=2, nlist=2, nprobe=1):
        """Demonstrate IVF (Inverted File) approximate nearest neighbor search."""
        print("\n=== IVF Vector Store Demo ===")

        # Create and populate IVF store
        store = TinyIVF(nlist=nlist, iters=10, seed=42)

        # Prepare items for IVF: (id, vector, metadata)
        items = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = {
                "title": f"Document {i+1}",
                "text": text,
                "category": "action" if "action" in text.lower() else "other"
            }
            items.append((f"doc{i+1}", vector.tolist(), metadata))

        # Fit the IVF index (builds clusters)
        store.fit(items)

        # Search
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        matches = store.knn(query_vector, k=k, nprobe=nprobe)

        print(f"Query: '{query}'")
        print(f"IVF Config: nlist={nlist}, nprobe={nprobe}")
        print("Top matches (id, score, metadata):", matches)
        return matches

    def demo_ivf_filtered_search(self, texts, embeddings, query="action movie", category_filter="action", k=2, nlist=2, nprobe=2):
        """Demonstrate IVF search with metadata filtering."""
        print("\n=== IVF Filtered Search Demo ===")

        # Create and populate IVF store
        store = TinyIVF(nlist=nlist, iters=10, seed=42)

        # Prepare items for IVF
        items = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = {
                "title": f"Document {i+1}",
                "text": text,
                "category": "action" if "action" in text.lower() else "other"
            }
            items.append((f"doc{i+1}", vector.tolist(), metadata))

        # Fit the IVF index
        store.fit(items)

        # Search with filter
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        filter_fn = lambda meta: meta.get("category") == category_filter
        matches = store.knn(query_vector, k=k, nprobe=nprobe, filter_fn=filter_fn)

        print(f"Query: '{query}' (filtered by category: {category_filter})")
        print(f"IVF Config: nlist={nlist}, nprobe={nprobe}")
        print("Filtered matches (id, score, metadata):", matches)
        return matches

    def demo_ivf_performance_comparison(self, texts, embeddings, query="explosions and criminals", k=2):
        """Compare IVF search with different nprobe values."""
        print("\n=== IVF Performance Comparison Demo ===")

        # Create and populate IVF store
        store = TinyIVF(nlist=3, iters=10, seed=42)

        items = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = {
                "title": f"Document {i+1}",
                "text": text,
                "category": "action" if "action" in text.lower() else "other"
            }
            items.append((f"doc{i+1}", vector.tolist(), metadata))

        store.fit(items)
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()

        print(f"Query: '{query}'")
        print("Comparing different nprobe values:")

        for nprobe in [1, 2, 3]:
            matches = store.knn(query_vector, k=k, nprobe=nprobe)
            print(f"  nprobe={nprobe}: {matches}")

        return matches

    def run_all_demos(self):
        """Run all demonstrations."""
        # Encode texts
        texts, embeddings = self.encode_texts()

        # Display embeddings
        self.display_embeddings(texts, embeddings)

        # Run demos
        self.demo_basic_vector_store(texts, embeddings)
        self.demo_metadata_vector_store(texts, embeddings)
        self.demo_filtered_search(texts, embeddings)
        self.demo_ivf_vector_store(texts, embeddings)
        self.demo_ivf_filtered_search(texts, embeddings)
        self.demo_ivf_performance_comparison(texts, embeddings)


def main():
    """Main function to run all vector store demonstrations."""
    runner = VectorStoreRunner()
    runner.run_all_demos()


if __name__ == '__main__':
    main()
