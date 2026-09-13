import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def build_faiss_index(chunk_text, model):
    text_chunks = [chunk[0] for chunk in chunk_text]

    embeddings = model.encode(text_chunks)

    embeddings = np.asarray(embeddings,dtype="float32")

    # Normalize embeddings so inner product becomes
    # equivalent to cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def dense_retrieval(chunk_text, query, model, index, k):

    query_embedding = model.encode([query])

    query_embedding = np.asarray(query_embedding,dtype="float32")

    faiss.normalize_L2(query_embedding)

    k = min(k, len(chunk_text))

    distances, indices = index.search(query_embedding,k)

    return indices[0], distances[0]