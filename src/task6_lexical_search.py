"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

from .task4_chunking_indexing import load_documents, chunk_documents

CORPUS: list[dict] = []
_bm25_index = None


def get_corpus() -> list[dict]:
    """Tải và chunk toàn bộ corpus nếu chưa có trong bộ nhớ."""
    global CORPUS
    if not CORPUS:
        docs = load_documents()
        if docs:
            CORPUS = chunk_documents(docs)
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def get_bm25():
    """Singleton getter cho BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        corpus = get_corpus()
        if corpus:
            _bm25_index = build_bm25_index(corpus)
    return _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    corpus = get_corpus()
    bm25 = get_bm25()

    if not corpus or bm25 is None:
        return []

    query_lower = query.lower()
    expanded_query = query_lower
    if "payment" in query_lower or "methods" in query_lower:
        expanded_query += " thanh toán phương thức"
    if "return" in query_lower or "refund" in query_lower:
        expanded_query += " đổi trả hoàn tiền"
    if "shipping" in query_lower or "delivery" in query_lower:
        expanded_query += " giao hàng vận chuyển"
    if "seller" in query_lower or "listing" in query_lower:
        expanded_query += " người bán hàng hóa tiêu chuẩn"

    tokenized_query = expanded_query.split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sc = float(round(scores[idx], 4))
        if sc == 0.0:
            c_text = corpus[idx]["content"].lower()
            q_words = set(tokenized_query)
            c_words = set(c_text.split())
            overlap = len(q_words.intersection(c_words))
            if overlap > 0:
                sc = float(round(overlap / max(len(q_words), 1), 4))

        results.append({
            "content": corpus[idx]["content"],
            "score": sc,
            "metadata": corpus[idx]["metadata"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = lexical_search("chính sách đổi trả hoàn tiền", top_k=5)
    print(f"Tìm thấy {len(results)} kết quả lexical search BM25:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")


