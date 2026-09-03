# Now we take those 4 vectors and put them into a FAISS index

# embeddings
#    ↓
# FAISS index
#    ↓
# query embedding
#    ↓
# search
#    ↓
# Top-K results


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# load embedding model 
model = SentenceTransformer('all-MiniLM-L6-v2')

# document chunks
texts = [
    "The quick brown fox jumps over the lazy dog.", 
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold."]

# convert document chunks to embeddings
embeddings = model.encode(texts)

# FAISS expects float32
embeddings = np.array(embeddings).astype('float32')

faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1] # Every vector I am going to store has 384 numbers
index = faiss.IndexFlatIP(dimension) # This creates a FAISS index that can search our vectors

index.add(embeddings) # Add our vectors to the index

print("Number of vectors in FAISS:", index.ntotal)

query = "What is FAISS used for?"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype('float32')
faiss.normalize_L2(query_embedding)

k = 2
distances , indices = index.search(query_embedding, k)

for rank , idx in enumerate(indices[0]):
    print(f"Rank {rank + 1}: Document index {idx}, Distance: {distances[0][rank]}")