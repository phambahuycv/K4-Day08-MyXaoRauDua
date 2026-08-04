"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf,docx]"
    # Cần extra [pdf,docx] để convert đủ hai định dạng. Module này vẫn có bộ đọc
    # DOCX thuần Python làm fallback nếu MarkItDown thiếu dependency tùy chọn.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_METADATA = {
    "chinh-sach-thanh-toan": {
        "doc_id": "fpt-payment-policy",
        "title": "Chính Sách Thanh Toán Và Bảo Mật Giao Dịch FPT Shop",
        "customer_role": "buyer",
        "category": "payment",
        "source_url": "https://fptshop.com.vn/chinh-sach-thanh-toan",
    },
    "chinh-sach-giao-hang": {
        "doc_id": "fpt-shipping-policy",
        "title": "Chính Sách Giao Hàng Và Đồng Kiểm FPT Shop",
        "customer_role": "buyer",
        "category": "shipping",
        "source_url": "https://fptshop.com.vn/chinh-sach-giao-hang",
    },
    "chinh-sach-doi-san-pham": {
        "doc_id": "fpt-return-policy",
        "title": "Chính Sách Đổi Sản Phẩm Và Hoàn Tiền FPT Shop",
        "customer_role": "buyer",
        "category": "refund",
        "source_url": "https://fptshop.com.vn/chinh-sach-doi-tra",
    },
}


def _extract_docx_text(filepath: Path) -> str:
    """Đọc text DOCX trực tiếp để Task 3 vẫn chạy khi thiếu MarkItDown."""
    with zipfile.ZipFile(filepath) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        if text.strip():
            style = paragraph.find(f"{namespace}pPr/{namespace}pStyle")
            if style is not None and style.attrib.get(f"{namespace}val", "").lower().startswith("heading"):
                text = f"# {text.strip()}"
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def _metadata_header(stem: str) -> str:
    metadata = LEGAL_METADATA.get(stem)
    if not metadata:
        return ""
    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    lines.extend(["language: vi", "retrieved_at: 2026-08-03", 'document_version: "2026"', "---", ""])
    return "\n".join(lines)


def _with_metadata(stem: str, content: str) -> str:
    if content.lstrip().startswith("---"):
        return content
    return _metadata_header(stem) + content


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown() if HAS_MARKITDOWN else None

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            converted = False
            if md:
                try:
                    result = md.convert(str(filepath))
                    if result and result.text_content:
                        output_path.write_text(
                            _with_metadata(filepath.stem, result.text_content),
                            encoding="utf-8",
                        )
                        converted = True
                except Exception as e:
                    print(f"  [WARN] MarkItDown warning: {e}")
            
            if not converted and filepath.suffix.lower() in (".docx", ".doc"):
                try:
                    output_path.write_text(
                        _with_metadata(filepath.stem, _extract_docx_text(filepath)),
                        encoding="utf-8",
                    )
                    converted = True
                except Exception as ex:
                    print(f"  [WARN] DOCX extract failed: {ex}")

            if not converted:
                # Check for original markdown in k4_ecommerce or extract text
                k4_md = Path(__file__).parent.parent / "data" / "k4_ecommerce" / f"{filepath.stem}.md"
                if k4_md.exists():
                    output_path.write_text(k4_md.read_text(encoding="utf-8"), encoding="utf-8")
                else:
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(filepath))
                        text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        output_path.write_text(_with_metadata(filepath.stem, text), encoding="utf-8")
                    except Exception as ex:
                        print(f"  [WARN] Fallback extract failed: {ex}")
            print(f"  [OK] Saved: {output_path}")



def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            raw_content = data.get("content_markdown", "")
            if raw_content.lstrip().startswith("---"):
                content = raw_content
            else:
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
                content = header + raw_content
            output_path.write_text(content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output tai:", OUTPUT_DIR)



if __name__ == "__main__":
    convert_all()

