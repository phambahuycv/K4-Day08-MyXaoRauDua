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

import re

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None

TOKEN_ALIASES = {
    "payment": ("thanh", "toán", "phương", "thức"),
    "payments": ("thanh", "toán", "phương", "thức"),
    "methods": ("phương", "thức"),
    "method": ("phương", "thức"),
    "return": ("trả", "đổi"),
    "refund": ("hoàn", "tiền"),
    "seller": ("người", "bán", "nhà", "bán"),
    "listing": ("đăng", "bán", "niêm", "yết"),
    "regulations": ("quy", "định"),
    "policy": ("chính", "sách"),
    "shipping": ("giao", "hàng", "vận", "chuyển"),
    "privacy": ("riêng", "tư", "bảo", "mật"),
    "order": ("đơn", "hàng"),
    "tracking": ("theo", "dõi", "vận", "chuyển"),
    "guide": ("hướng", "dẫn", "quy", "trình"),
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(TOKEN_ALIASES.get(token, ()))
    return expanded


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    global CORPUS, _BM25_INDEX
    CORPUS = list(corpus)
    if not CORPUS:
        _BM25_INDEX = None
        return None
    _BM25_INDEX = BM25Okapi([_tokenize(doc["content"]) for doc in CORPUS])
    return _BM25_INDEX


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
    if not query or top_k <= 0:
        return []
    _ensure_index()
    if _BM25_INDEX is None:
        return []
    scores = _BM25_INDEX.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: float(scores[index]), reverse=True
    )
    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[index]["content"],
                "score": score,
                "metadata": CORPUS[index]["metadata"],
                "source": "lexical",
            }
        )
        if len(results) >= top_k:
            break
    return results


def _ensure_index() -> None:
    global _BM25_INDEX
    if _BM25_INDEX is not None:
        return
    try:
        from .task4_chunking_indexing import chunk_documents, load_documents

        build_bm25_index(chunk_documents(load_documents()))
    except (ImportError, FileNotFoundError):
        return


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
