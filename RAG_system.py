import chunking
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

pdf_path = "sample.pdf"

def dense_retrieval(chunk_text , query , k) :

    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_chunk = []
    for i in range(len(chunk_text)) :
        text_chunk.append(chunk_text[i][0])
    print("Embedding model loaded and text chunks created.")
    embeddings = model.encode(text_chunk)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)

    distances , indices = index.search(query_embedding, k)

    for rank , idx in enumerate(indices[0]):
        print(f"Rank {rank + 1}: Document index {idx}, Distance: {distances[0][rank]} , Text: {chunk_text[idx][0]} , page: {chunk_text[idx][3]}")

    return indices[0], distances[0]