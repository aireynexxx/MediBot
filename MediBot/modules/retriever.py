from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDINGS_FILE = "data/embeddings.index"
METADATA_FILE = "data/texts.pkl"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(EMBEDDINGS_FILE)
with open(METADATA_FILE, "rb") as f:
    texts = pickle.load(f)


def retrieve_similar(query, k=8):
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), k)
    candidates = [texts[i] for i in I[0]]

    # Re-encode candidates for better comparison
    candidate_vecs = model.encode(candidates)
    query_vec = model.encode([query])[0]

    # Compute cosine similarity manually for re-ranking
    similarities = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
    )

    # Re-rank and return top 4
    ranked = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:4]]
