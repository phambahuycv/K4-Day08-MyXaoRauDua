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
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_DIR = Path(__file__).parent.parent
DOCUMENT_REGISTRY = PROJECT_DIR / "data" / "pageindex_documents.json"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    pdf_files = sorted((PROJECT_DIR / "data" / "landing").rglob("*.pdf"))
    registry = {}
    for pdf_file in pdf_files:
        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}")
        registry[pdf_file.name] = doc_id
        print(f"  Uploaded: {pdf_file.name} -> {doc_id}")
    DOCUMENT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def _document_ids() -> list[str]:
    configured = [value.strip() for value in os.getenv("PAGEINDEX_DOC_IDS", "").split(",") if value.strip()]
    if configured:
        return configured
    if DOCUMENT_REGISTRY.exists():
        return list(json.loads(DOCUMENT_REGISTRY.read_text(encoding="utf-8")).values())
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY or not query.strip() or top_k <= 0:
        return []
    doc_ids = _document_ids()
    if not doc_ids:
        return []
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results = []
    for doc_id in doc_ids:
        response = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = response.get("retrieval_id") or response.get("id")
        if not retrieval_id:
            continue
        retrieval = {}
        for _ in range(30):
            retrieval = client.get_retrieval(retrieval_id)
            status = str(retrieval.get("status", "")).lower()
            if status in {"completed", "success", "ready"} or retrieval.get("retrieved_nodes"):
                break
            if status in {"failed", "error"}:
                break
            time.sleep(1)
        for node in retrieval.get("retrieved_nodes", []):
            groups = node.get("relevant_contents", [])
            for group in groups:
                if isinstance(group, dict):
                    group = [group]
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if content:
                        results.append({
                            "content": content,
                            "score": 1.0 / (len(results) + 1),
                            "metadata": {
                                "source": str(doc_id),
                                "section": item.get("section_title", ""),
                                "type": "pageindex",
                            },
                            "source": "pageindex",
                        })
                        if len(results) >= top_k:
                            return results
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
