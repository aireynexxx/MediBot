from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os
import pickle

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDINGS_FILE = "data/embeddings.index"
METADATA_FILE = "data/texts.pkl"

def embed_and_save():
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_csv("data/medidata.csv")

    texts = df["transcription"].dropna().tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, EMBEDDINGS_FILE)

    # Save text metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(texts, f)

    print("Done embedding.")
    print(f"Saved FAISS index to: {EMBEDDINGS_FILE}")
    print(f"Saved texts to: {METADATA_FILE}")

if __name__ == "__main__":
    embed_and_save()
