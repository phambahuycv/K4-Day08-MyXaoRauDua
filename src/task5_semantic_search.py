"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()


def semantic_search(
    query: str, top_k: int = 10, customer_role: str | None = None
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or top_k <= 0:
        return []
    try:
        from .task4_chunking_indexing import get_collection, get_embedding_model

        collection = get_collection()
        if collection.count() == 0:
            return []
        model = get_embedding_model()
        search_text = _hyde_query(query)
        query_vector = model.encode(
            [search_text], normalize_embeddings=True
        )[0].tolist()
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if customer_role in {"buyer", "seller"}:
            kwargs["where"] = {"customer_role": {"$in": [customer_role, "both"]}}
        results = collection.query(**kwargs)
    except (FileNotFoundError, ValueError, ImportError):
        return []

    output = []
    for document, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append(
            {
                "content": document,
                "score": round(score, 6),
                "metadata": metadata or {},
                "source": "semantic",
            }
        )
    return sorted(output, key=lambda item: item["score"], reverse=True)[:top_k]


def _hyde_query(query: str) -> str:
    """HyDE tùy chọn, mặc định tắt để tránh thêm một lượt gọi LLM mỗi query."""
    if os.getenv("HYDE_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return query
    gemini_key = os.getenv("GEMINI_API_KEY")
    try:
        from openai import OpenAI
        if gemini_key:
            response = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=15,
            ).chat.completions.create(
                model=os.getenv("LLM_MODEL") or "gemma-4-31b-it",
                messages=[
                    {
                        "role": "user",
                        "content": f"Viết một đoạn trả lời giả định ngắn cho câu hỏi: {query}",
                    }
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content or query
    except Exception:
        pass

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return query
    try:
        from openai import OpenAI
        response = OpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=15
        ).chat.completions.create(
            model="google/gemma-4-31b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": f"Viết một đoạn trả lời giả định ngắn cho câu hỏi: {query}",
                }
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content or query
    except Exception:
        return query


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
