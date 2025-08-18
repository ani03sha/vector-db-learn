from sentence_transformers import SentenceTransformer

from .tiny_vector_store import TinyVectorStore

model = SentenceTransformer('all-MiniLM-L6-v2')

# Some text data to encode
text_data = [
    "Die Hard is an action movie about a cop fighting terrorists in a skyscraper.",
    "Lethal Weapon is a buddy-cop action film with lots of explosions.",
    "Frozen is a Disney movie about two sisters and magical ice powers."
]

# Convert to vectors
embeddings = model.encode(text_data, normalize_embeddings=True)

for text, vector in zip(text_data, embeddings):
    print(text, "\nVector (first 5 dims):", vector[:5], "\n")

# Use these embeddings in TinyVectorStore
tiny_vector_store = TinyVectorStore()
for i, (text, vector) in enumerate(zip(text_data, embeddings)):
    tiny_vector_store.upsert(f"doc{i+1}", vector.tolist())

query = "explosions and criminals"
query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

matches = tiny_vector_store.knn(query_vector, k=2)
print("Top matches:", matches)