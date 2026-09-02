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
doc1_vec  = np.array([0.19, -0.48, 0.85])  # Highly relevant
doc2_vec  = np.array([-0.75, 0.12, -0.34]) # Irrelevant

print(f"Similarity (Query vs Doc 1): {cosine_similarity(query_vec, doc1_vec):.4f}")
print(f"Similarity (Query vs Doc 2): {cosine_similarity(query_vec, doc2_vec):.4f}")