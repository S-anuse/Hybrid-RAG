import json
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import bm25_library
import chunking


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FOLDER = "Sample"

CHUNK_SIZE = 1000
OVERLAP = 2

K_VALUES = [3, 5, 10]
MAX_K = max(K_VALUES)

RRF_K = 60

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# evaluation_dataset.json stores p7(1).pdf / p8(1).pdf
# but the actual files in Sample/ are p7.pdf / p8.pdf.
PDF_NAME_MAP = {
    "p7(1).pdf": "p7.pdf",
    "p8(1).pdf": "p8.pdf"
}


# ============================================================
# LOAD DATASET
# ============================================================

with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Total evaluation questions:", len(dataset))


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("\nLoading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading CrossEncoder...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)

print("Models loaded successfully.")


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def recall_at_k(ranked_chunk_ids, relevant_chunk_ids, k):
    """
    Recall@K =
    number of relevant chunks found in top K
    ------------------------------------------------
    total number of relevant chunks
    """

    relevant_set = set(relevant_chunk_ids)

    top_k = ranked_chunk_ids[:k]

    retrieved_relevant = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_set
    )

    if len(relevant_set) == 0:
        return 0.0

    return retrieved_relevant / len(relevant_set)


def reciprocal_rank(ranked_chunk_ids, relevant_chunk_ids):
    """
    Reciprocal Rank =
    1 / rank of the first relevant chunk

    Example:
    relevant chunk appears at rank 1 -> 1.0
    rank 2 -> 0.5
    rank 5 -> 0.2
    no relevant chunk -> 0.0
    """

    relevant_set = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):

        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

methods = [
    "FAISS",
    "BM25",
    "RRF",
    "RRF + CrossEncoder"
]


results = {
    method: {
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "mrr": []
    }
    for method in methods
}


# ============================================================
# PROCESS EACH PDF
# ============================================================

pdf_names = sorted(set(item["source"] for item in dataset))

print("\nNumber of PDFs:", len(pdf_names))


for pdf_name in pdf_names:

    actual_name = PDF_NAME_MAP.get(pdf_name, pdf_name)

    pdf_path = os.path.join(PDF_FOLDER, actual_name)

    print("\n" + "=" * 80)
    print("PDF:", actual_name)
    print("=" * 80)

    if not os.path.exists(pdf_path):
        print("ERROR: File not found:", pdf_path)
        continue


    # --------------------------------------------------------
    # QUESTIONS FOR THIS PDF
    # --------------------------------------------------------

    pdf_questions = [
        item
        for item in dataset
        if item["source"] == pdf_name
    ]

    print("Questions:", len(pdf_questions))


    # --------------------------------------------------------
    # CHUNK PDF ONCE
    # --------------------------------------------------------

    chunks = chunking.chunk_texts(
        pdf_path,
        CHUNK_SIZE,
        OVERLAP
    )

    print("Chunks:", len(chunks))


    # --------------------------------------------------------
    # EXTRACT CHUNK TEXTS
    # --------------------------------------------------------

    chunk_texts = [
        chunk[0]
        for chunk in chunks
    ]


    # ========================================================
    # BUILD FAISS INDEX ONCE
    # ========================================================

    print("Building FAISS index...")

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors so inner product behaves like
    # cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)


    # ========================================================
    # BUILD BM25 INDEX ONCE
    # ========================================================

    print("Building BM25 index...")

    bm25 = bm25_library.build_bm25_index(chunks)


    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for question_number, item in enumerate(pdf_questions, start=1):

        query = item["question"]

        relevant_chunk_ids = item["relevant_chunk_ids"]


        # ====================================================
        # 1. FAISS RETRIEVAL
        # ====================================================

        query_embedding = embedding_model.encode(
            [query]
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        faiss.normalize_L2(query_embedding)

        distances, indices = faiss_index.search(
            query_embedding,
            min(MAX_K, len(chunks))
        )

        dense_positions = indices[0].tolist()

        # FAISS returns array positions.
        # Convert those positions to our actual chunk IDs.
        dense_chunk_ids = [
            chunks[position][2]
            for position in dense_positions
        ]


        # ====================================================
        # 2. BM25 RETRIEVAL
        # ====================================================

        query_tokens = query.lower().split()

        bm25_scores = bm25.get_scores(query_tokens)

        bm25_positions = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        bm25_positions = bm25_positions[:MAX_K]

        bm25_chunk_ids = [
            chunks[position][2]
            for position in bm25_positions
        ]


        # ====================================================
        # 3. RRF FUSION
        # ====================================================

        rrf_scores = {}

        # Dense ranking
        for rank, position in enumerate(dense_positions):

            rrf_scores[position] = (
                rrf_scores.get(position, 0)
                + 1 / (RRF_K + rank + 1)
            )


        # BM25 ranking
        for rank, position in enumerate(bm25_positions):

            rrf_scores[position] = (
                rrf_scores.get(position, 0)
                + 1 / (RRF_K + rank + 1)
            )


        # Sort by RRF score
        rrf_positions = sorted(
            rrf_scores.keys(),
            key=lambda position: rrf_scores[position],
            reverse=True
        )

        rrf_positions = rrf_positions[:MAX_K]

        rrf_chunk_ids = [
            chunks[position][2]
            for position in rrf_positions
        ]


        # ====================================================
        # 4. CROSSENCODER RERANKING
        # ====================================================

        candidate_texts = [
            chunks[position][0]
            for position in rrf_positions
        ]

        pairs = [
            [query, text]
            for text in candidate_texts
        ]

        rerank_scores = reranker.predict(pairs)

        reranked_positions = sorted(
            range(len(rerank_scores)),
            key=lambda i: rerank_scores[i],
            reverse=True
        )

        reranked_positions = reranked_positions[:MAX_K]

        reranked_chunk_ids = [
            rrf_chunk_ids[position]
            for position in reranked_positions
        ]


        # ====================================================
        # CALCULATE METRICS FOR EACH METHOD
        # ====================================================

        rankings = {
            "FAISS": dense_chunk_ids,
            "BM25": bm25_chunk_ids,
            "RRF": rrf_chunk_ids,
            "RRF + CrossEncoder": reranked_chunk_ids
        }


        for method, ranked_chunk_ids in rankings.items():

            for k in K_VALUES:

                score = recall_at_k(
                    ranked_chunk_ids,
                    relevant_chunk_ids,
                    k
                )

                results[method][f"recall@{k}"].append(score)


            rr = reciprocal_rank(
                ranked_chunk_ids,
                relevant_chunk_ids
            )

            results[method]["mrr"].append(rr)


        print(
            f"Question {question_number:2d}/{len(pdf_questions)} done"
        )


# ============================================================
# FINAL AGGREGATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 80)


print(
    f"{'Method':<22}"
    f"{'Recall@3':>12}"
    f"{'Recall@5':>12}"
    f"{'Recall@10':>13}"
    f"{'MRR':>10}"
)

print("-" * 80)


final_results = {}


for method in methods:

    recall3 = np.mean(
        results[method]["recall@3"]
    )

    recall5 = np.mean(
        results[method]["recall@5"]
    )

    recall10 = np.mean(
        results[method]["recall@10"]
    )

    mrr = np.mean(
        results[method]["mrr"]
    )


    final_results[method] = {
        "Recall@3": recall3,
        "Recall@5": recall5,
        "Recall@10": recall10,
        "MRR": mrr
    }


    print(
        f"{method:<22}"
        f"{recall3:>12.4f}"
        f"{recall5:>12.4f}"
        f"{recall10:>13.4f}"
        f"{mrr:>10.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


print("\nResults saved to: evaluation_results.json")