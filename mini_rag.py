import numpy as np
def cosine_similarity(u : np.ndarray, v : np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters
    ----------
    u : np.ndarray
        First vector.
    v : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity between u and v.
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return float(dot_product / (norm_u * norm_v))

query_vec = np.array([0.21, -0.45, 0.89])
documents = {
    "doc1": np.array([1, 2, 3]),
    "doc2": np.array([10, 1, 0]),
    "doc3": np.array([1, 3, 2]),
    "doc4": np.array([0, 1, 0])
}

results = [] 
for doc_id , doc_vec in documents.items():
    similarity = cosine_similarity(query_vec, doc_vec)
    results.append((doc_id, similarity))

results.sort(key=lambda x : x[1] , reverse=True)

for doc_id, similarity in results:
    print(f"Similarity (Query vs {doc_id}): {similarity:.4f}")

# top 2 documents
top_k = 2 
top_results = results[:top_k]
for doc_id, similarity in top_results:
    print(f"Top {top_k} Similarity (Query vs {doc_id}): {similarity:.4f}")