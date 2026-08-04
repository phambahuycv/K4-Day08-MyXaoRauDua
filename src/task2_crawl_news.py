"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
K4_DIR = Path(__file__).parent.parent / "data" / "k4_ecommerce"

ARTICLE_URLS = [
    "https://seller-vn.tiktok.com/university/essay?knowledge_id=2901402355762946&lang=vi-VN",
    "https://seller-vn.tiktok.com/university/essay?default_language=vi-VN&knowledge_id=6837776050308866&lang=vi-VN",
    "https://seller-vn.tiktok.com/university/essay?default_language=vi-VN&knowledge_id=7045464018339600&lang=vi-VN",
    "https://seller-vn.tiktok.com/university/essay?knowledge_id=8692722068424464&lang=vi-VN",
    "https://seller-vn.tiktok.com/university/essay?article_type=agreement&default_language=vi-VN&knowledge_id=6837773789234946&lang=vi-VN",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def _html_to_markdown(raw_html: str) -> str:
    """Trích xuất text cơ bản khi nguồn không trả về Markdown."""
    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", without_scripts)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def crawl_article(url: str) -> dict:
    """Crawl một URL và trả về schema chuẩn của Task 2."""
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGLab/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    content = _html_to_markdown(response.text)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.S | re.I)
    title = _html_to_markdown(title_match.group(1)) if title_match else url
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().date().isoformat(),
        "content_markdown": content,
    }


def pack_news_articles():
    """Đóng gói bài viết từ k4_ecommerce thành các file JSON trong data/landing/news/."""
    setup_directory()

    existing = sorted(DATA_DIR.glob("*.json"))
    if len(existing) >= 5:
        print(f"[OK] Da co san {len(existing)} bai JSON, giu nguyen du lieu da crawl.")
        return

    # Read sources.csv metadata mapping
    sources_map = {}
    csv_file = K4_DIR / "sources.csv"
    if csv_file.exists():
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_name = Path(row["file_path"]).name
                sources_map[doc_name] = row

    tiktok_urls = ARTICLE_URLS

    news_files = [
        "tiktok-community-guidelines.md",
        "tiktok-terms-of-service.md",
        "tiktok-virtual-items.md",
        "tiktok-copyright-policy.md",
        "tiktok-privacy-policy.md",
    ]

    for i, (file_name, url) in enumerate(zip(news_files, tiktok_urls), 1):
        file_path = K4_DIR / file_name
        meta = sources_map.get(file_name, {})
        if file_path.exists():
            article_data = {
                "url": url,
                "title": meta.get("title", file_name.replace(".md", "").replace("-", " ").title()),
                "date_crawled": meta.get("retrieved_at", datetime.now().date().isoformat()),
                "content_markdown": file_path.read_text(encoding="utf-8"),
            }
        else:
            try:
                article_data = crawl_article(url)
            except requests.RequestException as exc:
                print(f"[WARN] Khong crawl duoc {url}: {exc}")
                continue

        output_filename = f"article_{i:02d}.json"
        output_filepath = DATA_DIR / output_filename
        output_filepath.write_text(json.dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Da tao JSON: {output_filepath.name} ({output_filepath.stat().st_size} bytes)")




if __name__ == "__main__":
    pack_news_articles()

