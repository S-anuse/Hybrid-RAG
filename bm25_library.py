from rank_bm25 import BM25Okapi
import chunking

# BM25Okapi
#     ↓
# build BM25 index from documents
#     ↓
# query
#     ↓
# retrieve ranked documents


def build_bm25_index(chunk_text) :

    documents = []
    for i in range(len(chunk_text)) :
        documents.append(chunk_text[i][0])
    

    tokenized_corpus = [doc.lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25