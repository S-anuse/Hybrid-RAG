import chunking
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

pdf_path = "sample.pdf"
chunks = chunking.chunk_text(pdf_path, 500, 1)

model = SentenceTransformer('all-MiniLM-L6-v2')
text_chunk = []
for i in range(len(chunks)) :
    text_chunk.append(chunks[i][0])
print("Embedding model loaded and text chunks created.")
embeddings = model.encode(text_chunk)
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

query = input("Enter your query: ")
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype('float32')
faiss.normalize_L2(query_embedding)

k = input("Enter the number of top results to retrieve: ")
k = int(k)

distances , indices = index.search(query_embedding, k)

for rank , idx in enumerate(indices[0]):
    print(f"Rank {rank + 1}: Document index {idx}, Distance: {distances[0][rank]} , Text: {chunks[idx][0]} , source: {pdf_path} , page: {chunks[idx][3]}")
