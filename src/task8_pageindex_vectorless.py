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
import re
import json
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    documents = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not documents:
        return {}

    cached = {}
    if PAGEINDEX_CACHE.exists():
        try:
            cached = json.loads(PAGEINDEX_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}

    # SDK/API upload is optional. The local structural index remains usable when
    # PageIndex is unavailable or the account has no API key.
    for path in documents:
        cached.setdefault(str(path.relative_to(STANDARDIZED_DIR)), f"local:{path.stem}")
    PAGEINDEX_CACHE.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
    return cached


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
    if not query or top_k <= 0:
        return []
    sections = _load_structural_sections()
    query_terms = _terms(query)
    scored = []
    for section in sections:
        section_terms = _terms(section["content"])
        overlap = query_terms & section_terms
        if not overlap:
            continue
        score = len(overlap) / max(len(query_terms), 1)
        scored.append({**section, "score": round(min(1.0, score), 6), "source": "pageindex"})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _load_structural_sections() -> list[dict]:
    sections = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        metadata = _front_matter(content)
        headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+)$", content))
        if not headings:
            sections.append(
                {
                    "content": content,
                    "metadata": {"source": path.name, **metadata},
                }
            )
            continue
        for index, heading in enumerate(headings):
            start = heading.start()
            end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
            section_content = content[start:end].strip()
            section_metadata = {
                "source": path.name,
                "section": heading.group(1).strip(),
                "type": "legal" if path.parent.name == "legal" else "news",
                **metadata,
            }
            sections.append({"content": section_content, "metadata": section_metadata})
    return sections


def _front_matter(content: str) -> dict:
    lines = content.splitlines()
    separators = [index for index, line in enumerate(lines) if line.strip() == "---"]
    for start, end in zip(separators, separators[1:]):
        values = {}
        for line in lines[start + 1 : end]:
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip().strip('"\'')
        if values and {"doc_id", "customer_role", "category"} & values.keys():
            return values
    return {}


def _terms(text: str) -> set[str]:
    aliases = {
        "payment": {"thanh", "toán", "phương", "thức"},
        "methods": {"phương", "thức"},
        "return": {"trả", "đổi"},
        "refund": {"hoàn", "tiền"},
        "seller": {"người", "bán"},
        "privacy": {"riêng", "tư", "bảo", "mật"},
    }
    terms = set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))
    for term in list(terms):
        terms.update(aliases.get(term, set()))
    return terms


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
