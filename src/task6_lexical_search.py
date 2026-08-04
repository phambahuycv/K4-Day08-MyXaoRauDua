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
import re
import unicodedata

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25 = None


def tokenize(text: str) -> list[str]:
    """Unicode-aware lowercase tokenizer suitable for Vietnamese and English."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """Build the lexical corpus from the same chunks used by dense retrieval."""
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        return None
    return BM25Okapi([tokenize(doc.get("content", "")) for doc in corpus])


def refresh_index() -> int:
    """Reload standardized documents and rebuild the process-local BM25 index."""
    global CORPUS, _BM25
    CORPUS = load_corpus()
    _BM25 = build_bm25_index(CORPUS)
    return len(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _BM25
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    if _BM25 is None:
        refresh_index()
    if _BM25 is None:
        return []
    scores = _BM25.get_scores(tokenize(query))
    ranked = sorted(range(len(CORPUS)), key=lambda index: scores[index], reverse=True)
    results = []
    for index in ranked:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append({
            "content": CORPUS[index]["content"],
            "score": score,
            "metadata": CORPUS[index].get("metadata", {}),
        })
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
