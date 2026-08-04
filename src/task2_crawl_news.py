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

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/188931",
    "https://help.shopee.vn/portal/4/article/79233",
    "https://help.shopee.vn/portal/4/article/189473",
    "https://help.shopee.vn/portal/4/article/79377",
    "https://help.shopee.vn/portal/4/article/79526",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    if not url.startswith(("http://", "https://")):
        raise ValueError("url must use http or https")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if not getattr(result, "success", True):
            raise RuntimeError(getattr(result, "error_message", "Crawl failed"))
        markdown = getattr(result, "markdown", "") or ""
        if hasattr(markdown, "raw_markdown"):
            markdown = markdown.raw_markdown
        metadata = getattr(result, "metadata", {}) or {}
    except Exception:
        # Public help pages also expose server-rendered text, which provides a
        # deterministic fallback when the optional Playwright browser is absent.
        import requests
        from bs4 import BeautifulSoup

        response = await asyncio.to_thread(
            requests.get, url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else "Unknown"
        markdown = "\n\n".join(
            line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
        )
        metadata = {"title": title}
    return {
        "url": url,
        "title": metadata.get("title") or "Unknown",
        "date_crawled": datetime.now().astimezone().isoformat(),
        "content_markdown": str(markdown),
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
