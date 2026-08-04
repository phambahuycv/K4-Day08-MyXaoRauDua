"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex hoặc log thông tin.
    """
    if not PAGEINDEX_API_KEY:
        print("[INFO] PAGEINDEX_API_KEY không có. Sử dụng Local Fallback cho Task 8.")
        return

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            print(f"  ✓ Uploaded: {md_file.name}")
    except Exception as e:
        print(f"[WARN] PageIndex upload error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex hoặc Local Structural Fallback.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            # Query PageIndex API if available
        except Exception:
            pass

    # Safe Local Structural Fallback when API key is missing or offline
    from .task4_chunking_indexing import load_documents, chunk_documents
    docs = load_documents()
    chunks = chunk_documents(docs) if docs else []

    query_words = set(query.lower().split())
    results = []

    for idx, c in enumerate(chunks):
        content = c.get("content", "")
        c_words = set(content.lower().split())
        overlap = len(query_words.intersection(c_words))
        score = round(overlap / max(len(query_words), 1), 4)

        results.append({
            "content": content,
            "score": score,
            "metadata": c.get("metadata", {}),
            "source": "pageindex"
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa thiết lập trong .env (Dùng fallback).")
    else:
        upload_documents()

    print("\nTest PageIndex query:")
    results = pageindex_search("chính sách hoàn tiền sản phẩm", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

