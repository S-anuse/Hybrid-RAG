import chunking , RAG_system , bm25_library 

chunk_size = 1000
overlap = 2
pdf_path = "sample.pdf"
chunk_text = chunking.chunk_texts(pdf_path , chunk_size , overlap)
 
bm25 = bm25_library.build_bm25_index(chunk_text)
query = input("Enter your query: ")
k = int(input("Enter number of results: "))

dense_indices , dense_scores = RAG_system.dense_retrieval(chunk_text , query , k)

print("Dense indices:", dense_indices)
print("Dense scores:", dense_scores)

query_tokens = query.lower().split()
bm25_scores = bm25.get_scores(query_tokens)

bm25_indices = sorted(range(len(bm25_scores)) , key=lambda i : bm25_scores[i] , reverse=True)
bm25_indices = bm25_indices[:k]

print("BM25 scores:", bm25_scores)
print("BM25 indices:", bm25_indices)

rrf_scores = {}
rrf_k = 60
for rank , doc_id in enumerate(dense_indices) :
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

for rank , doc_id in enumerate(bm25_indices) :
    rrf_scores[doc_id] = rrf_scores.get(doc_id , 0) + 1 / (rrf_k + rank + 1)


hybrid_indices = sorted(rrf_scores , key=rrf_scores.get, reverse = True)
hybrid_indices = hybrid_indices[:k]

print("RRF scores:", rrf_scores)
print("Hybrid indices:", hybrid_indices)