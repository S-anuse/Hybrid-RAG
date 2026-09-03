# pretrained sentence-transformer embedding model

# PDF
#  ↓
# Text
#  ↓
# Chunks
#  ↓
# Embedding model
#  ↓
# One embedding per chunk
#  ↓
# FAISS

# Question
#    ↓
# Embedding model
#    ↓
# Query embedding
#    ↓
# FAISS searches all chunk embeddings
#    ↓
# Top-K relevant chunks

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold."]

embeddings = model.encode(texts)

print("Number of texts:", len(texts))
print("Number of embeddings:", embeddings.shape)
print("First embedding vector:", embeddings[0])