import os

import chunking
import RAG_system
import bm25_library

from google import genai
from sentence_transformers import CrossEncoder


CHUNK_SIZE = 1000
OVERLAP = 2
RRF_K = 60

RERANKER_MODEL_NAME = ("cross-encoder/ms-marco-MiniLM-L-6-v2")


pdf_path = input("Enter PDF path: ").strip()

if not os.path.exists(pdf_path):
    print("PDF not found:", pdf_path)
    exit()


print("\nChunking PDF...")

chunk_text = chunking.chunk_texts(pdf_path,CHUNK_SIZE,OVERLAP)

print("Total chunks:", len(chunk_text))


print("\nLoading embedding model...")

embedding_model = RAG_system.load_embedding_model()


print("Building FAISS index...")

faiss_index = RAG_system.build_faiss_index(chunk_text,embedding_model)


print("Building BM25 index...")

bm25 = bm25_library.build_bm25_index(chunk_text)


print("Loading CrossEncoder...")

reranker = CrossEncoder(RERANKER_MODEL_NAME)

print("\nRAG system ready.")


client = genai.Client()


while True:

    query = input("\nEnter your query (or type 'exit'): ").strip()

    if query.lower() == "exit":
        break

    k = int(input("Enter number of results: "))

    k = min(k, len(chunk_text))


    dense_indices, dense_scores = (
        RAG_system.dense_retrieval(
            chunk_text,
            query,
            embedding_model,
            faiss_index,
            k
        )
    )

    print("\nDense indices:", dense_indices)
    print("Dense scores:", dense_scores)


    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_indices = bm25_indices[:k]

    print("\nBM25 indices:", bm25_indices)


    rrf_scores = {}


    for rank, doc_id in enumerate(dense_indices):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (RRF_K + rank + 1)
        )


    for rank, doc_id in enumerate(bm25_indices):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (RRF_K + rank + 1)
        )


    hybrid_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    hybrid_indices = hybrid_indices[:k]

    print("\nRRF scores:", rrf_scores)
    print("Hybrid indices:", hybrid_indices)


    candidate_texts = [chunk_text[i][0] for i in hybrid_indices]

    pairs = [[query, text] for text in candidate_texts]

    rerank_scores = reranker.predict(pairs)

    reranked_positions = sorted(
        range(len(rerank_scores)),
        key=lambda i: rerank_scores[i],
        reverse=True
    )

    reranked_positions = (reranked_positions[:k])

    reranked_indices = [hybrid_indices[position] for position in reranked_positions]

    final_chunks = [chunk_text[i] for i in reranked_indices]


    print("\nRerank scores:", rerank_scores)
    print("Reranked indices:", reranked_indices)


    for rank, chunk in enumerate(final_chunks,start=1):

        print(f"\nRank {rank}:")
        print("Text:", chunk[0])
        print("Source:", chunk[1])
        print("Chunk ID:", chunk[2])
        print("Pages:", chunk[3])


    context = ""

    for rank, chunk in enumerate(final_chunks,start=1):

        context += f"""
[Chunk {rank}]
Source: {chunk[1]}
Chunk ID: {chunk[2]}
Pages: {chunk[3]}

{chunk[0]}
"""


    prompt = f"""
Answer the question using only the supplied context.

Rules:
1. Do not use information that is not present in the context.
2. If the answer is not present in the context, say that the context does not contain enough information.
3. Do not generate citations or source information.

CONTEXT:
{context}

QUESTION:
{query}
"""


    response = client.models.generate_content(model="gemini-3.6-flash",contents=prompt)


    print("\nAnswer:")
    print(response.text)


    citations = set()

    for chunk in final_chunks:

        source = chunk[1]

        pages = tuple(sorted(chunk[3]))

        citations.add((source, pages))


    print("\nSources:")

    for source, pages in citations:

        print(f"- {source}, Pages: {list(pages)}")