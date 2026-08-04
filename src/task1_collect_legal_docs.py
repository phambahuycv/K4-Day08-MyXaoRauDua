"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
from typing import Iterable

import requests

POLICY_DOCUMENTS = [
    ("https://help.shopee.vn/portal/4/article/77251", "returns-refund-policy-shopee.pdf"),
    ("https://help.shopee.vn/portal/4/article/79198", "payment-methods-shopee.pdf"),
    ("https://help.shopee.vn/portal/4/article/77244", "privacy-policy-shopee.pdf"),
]

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_file(url: str, filename: str, timeout: int = 60) -> Path:
    """Download a public PDF/DOCX file after validating its name and payload."""
    setup_directory()
    destination = DATA_DIR / Path(filename).name
    if destination.suffix.lower() not in {".pdf", ".doc", ".docx"}:
        raise ValueError("filename must end in .pdf, .doc, or .docx")
    response = requests.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    if len(response.content) < 1024:
        raise ValueError(f"Downloaded file is unexpectedly small: {url}")
    destination.write_bytes(response.content)
    return destination


def download_documents(documents: Iterable[tuple[str, str]]) -> list[Path]:
    """Download ``(url, filename)`` pairs and return their local paths."""
    return [download_file(url, filename) for url, filename in documents]


def capture_policy_as_pdf(url: str, filename: str, timeout: int = 60) -> Path:
    """Fetch a public HTML policy and preserve its readable text as a PDF."""
    from bs4 import BeautifulSoup
    from fpdf import FPDF

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    text = "\n".join(dict.fromkeys(lines))
    if len(text) < 500:
        raise ValueError(f"No substantial policy text found at {url}")

    setup_directory()
    destination = DATA_DIR / Path(filename).name
    pdf = FPDF()
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        pdf.add_font("Unicode", fname=str(font_path))
        font_name = "Unicode"
    else:
        font_name = "Helvetica"
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font(font_name, size=11)
    pdf.multi_cell(0, 6, f"Source: {url}\nRetrieved policy text\n\n{text}")
    pdf.output(str(destination))
    return destination


def collect_default_policies() -> list[Path]:
    """Collect the three official Shopee policy pages configured for this lab."""
    return [capture_policy_as_pdf(url, filename) for url, filename in POLICY_DOCUMENTS]


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.
#
# Ví dụ nếu có direct link:
#
# import requests
#
# def download_file(url: str, filename: str):
#     response = requests.get(url)
#     filepath = DATA_DIR / filename
#     filepath.write_bytes(response.content)
#     print(f"✓ Đã tải: {filepath}")
#
# Nếu trang là HTML thuần (không phải PDF sẵn), có thể convert nội dung text
# thành PDF đơn giản bằng thư viện fpdf2 (đã có trong requirements.txt).


if __name__ == "__main__":
    for path in collect_default_policies():
        print(f"Saved: {path}")
