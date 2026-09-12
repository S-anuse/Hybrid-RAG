import math

def BM25_retrieval(query , documents) :
    TF = {} 
    DF = {} 
    doc_len = []
    avg_doc_len = 0
    i = 0
    for doc in documents :
        doc_len.append(len(doc.split()))
        avg_doc_len += len(doc.split())
        for word in doc.split() :
            word = word.lower()
            if i not in TF :
                TF[i] = {}
            TF[i][word] = TF[i].get(word, 0) + 1
        i += 1

    avg_doc_len = avg_doc_len / len(documents)
            
    for q in query.split() :
        q = q.lower()
        DF[q] = 0
        for doc in documents :
            if q in [word.lower() for word in doc.split()]:
                DF[q] += 1

    return DF , TF , avg_doc_len , doc_len


documents = [
    "machine learning is useful",
    "deep learning is powerful",
    "machine learning models learn patterns"
]
BM25_retrieval_results = BM25_retrieval("machine learning", documents)
print(BM25_retrieval_results)

query = "machine learning"

k1 = 1.2
b= 0.75
scores = {}
avgdl = BM25_retrieval_results[2]
for word in query.split() :
    word = word.lower()
    idf = math.log(1 + (len(documents) - BM25_retrieval_results[0][word] + 0.5) / (BM25_retrieval_results[0][word] + 0.5))
    for doc_id in BM25_retrieval_results[1] :
        tf = BM25_retrieval_results[1][doc_id].get(word, 0)
        doc_len = BM25_retrieval_results[3][doc_id]
        contribution = idf * ((tf * (k1+1)) / (tf + k1 * (1  - b + b * (doc_len / avgdl))))
        scores[doc_id] = scores.get(doc_id, 0) + contribution


for doc_id, score in scores.items():
    print("Document", doc_id, ":", score)