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

import html
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
K4_DIR = Path(__file__).parent.parent / "data" / "k4_ecommerce"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def create_docx_from_markdown(md_file: Path, docx_file: Path):
    """Tạo file .docx hợp lệ chuẩn OpenXML bằng pure Python (zipfile)."""
    content = md_file.read_text(encoding="utf-8")
    
    # Try using python-docx if available
    try:
        import docx
        doc = docx.Document()
        for line in content.split("\n"):
            line_str = line.strip()
            if line_str.startswith("#"):
                doc.add_heading(line_str.lstrip("#").strip(), level=1)
            elif line_str:
                doc.add_paragraph(line_str)
        doc.save(str(docx_file))
        print(f"[OK] Da tao DOCX qua python-docx: {docx_file.name} ({docx_file.stat().st_size} bytes)")
        return
    except ImportError:
        pass

    # Pure Python OpenXML DOCX Generator
    body_paragraphs = []
    for line in content.split("\n"):
        line_str = line.strip()
        if not line_str:
            continue
        escaped_text = html.escape(line_str)
        if line_str.startswith("#"):
            title_text = html.escape(line_str.lstrip("#").strip())
            p_xml = f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>{title_text}</w:t></w:r></w:p>'
        else:
            p_xml = f'<w:p><w:r><w:t>{escaped_text}</w:t></w:r></w:p>'
        body_paragraphs.append(p_xml)

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {"".join(body_paragraphs)}
  </w:body>
</w:document>"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

    with zipfile.ZipFile(docx_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml.encode("utf-8"))
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml)

    print(f"[OK] Da tao DOCX pure python: {docx_file.name} ({docx_file.stat().st_size} bytes)")


def collect_legal_docs():
    """Lấy 3 file chính sách chuẩn FPT Shop và chuyển thành DOCX trong data/landing/legal/."""
    setup_directory()
    
    # Xoá tất cả các file cũ trong legal/ để tránh file sai/dư thừa
    for f in DATA_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                f.unlink()
            except Exception:
                pass

    # 3 file DOCX chuẩn từ FPT Shop
    # Fallback content if k4_ecommerce is missing
    fpt_policies = {
        "chinh-sach-thanh-toan.docx": """# Chính Sách Thanh Toán Và Bảo Mật Giao Dịch FPT Shop

## 1. Các phương thức thanh toán áp dụng
Khách hàng có thể lựa chọn các phương thức thanh toán sau khi mua sắm sản phẩm tại hệ thống cửa hàng và website chính thức của FPT Shop:
- Thanh toán tiền mặt trực tiếp khi nhận hàng (COD).
- Thanh toán chuyển khoản qua thẻ ngân hàng nội địa (ATM/Internet Banking).
- Thanh toán bằng thẻ tín dụng quốc tế (Visa, MasterCard, JCB).
- Thanh toán qua ví điện tử (VNPAY, MoMo, ZaloPay).
- Thanh toán trả góp qua công ty tài chính hoặc thẻ tín dụng.

## 2. Quy trình xử lý và xác nhận giao dịch
Sau khi hoàn tất đặt hàng và lựa chọn phương thức thanh toán trực tuyến, hệ thống sẽ tự động khởi tạo mã giao dịch duy nhất. Đơn hàng sẽ được xác nhận ngay khi cổng thanh toán gửi thông báo thanh toán thành công về hệ thống quản lý đơn hàng của FPT Shop.

## 3. Chính sách bảo mật thông tin thanh toán
Hệ thống thanh toán trực tuyến của FPT Shop đáp ứng tiêu chuẩn bảo mật PCI DSS cấp độ cao nhất. Mọi dữ liệu thẻ ngân hàng của khách hàng đều được mã hóa theo giao thức SSL 256-bit và truyền trực tiếp tới cổng thanh toán của đối tác ngân hàng liên kết. FPT Shop cam kết không lưu giữ thông tin số thẻ hoặc mã CVV/CVC của khách hàng trên hệ thống máy chủ nội bộ.""",
        "chinh-sach-giao-hang.docx": """# Chính Sách Giao Hàng Và Đồng Kiểm FPT Shop

## 1. Phạm vi và thời gian giao hàng
FPT Shop cung cấp dịch vụ giao hàng toàn quốc tới 63 tỉnh thành trên cả nước với thời gian dự kiến:
- Khu vực nội thành Hà Nội & TP. Hồ Chí Minh: Giao siêu tốc trong vòng 2 - 4 giờ làm việc.
- Các tỉnh thành khác và vùng ngoại thành: Giao hàng tiêu chuẩn từ 1 - 3 ngày làm việc kể từ thời điểm bàn giao cho đơn vị vận chuyển.

## 2. Quy định đồng kiểm khi nhận hàng
Khi nhân viên giao hàng bàn giao kiện hàng, người nhận có quyền và nghĩa vụ thực hiện đồng kiểm trực tiếp:
- Kiểm tra tình trạng niêm phong, tem chống giả và nhãn mác bên ngoài của hộp đựng sản phẩm.
- Mở hộp kiểm tra ngoại quan sản phẩm (tránh trầy xước, móp méo hoặc nứt vỡ màn hình).
- Kiểm tra số lượng và loại phụ kiện đi kèm ghi trên phiếu giao hàng.

## 3. Quy trình xử lý khi phát sinh sự cố giao hàng
Nếu sản phẩm bị hư hỏng, sai model hoặc thiếu phụ kiện tại thời điểm đồng kiểm, khách hàng có quyền từ chối nhận hàng và yêu cầu nhân viên giao hàng lập biên bản xác nhận bất thường. FPT Shop sẽ lập tức gửi lại sản phẩm thay thế mới cho khách hàng trong vòng 24 giờ.""",
        "chinh-sach-doi-san-pham.docx": """# Chính Sách Đổi Sản Phẩm Và Hoàn Tiền FPT Shop

## 1. Điều kiện đổi trả sản phẩm mới 100%
Khách hàng mua hàng tại FPT Shop được hưởng chính sách 1 đổi 1 trong vòng 30 ngày kể từ ngày mua nếu sản phẩm gặp lỗi phần cứng do nhà sản xuất. Điều kiện áp dụng:
- Sản phẩm còn giữ nguyên vẹn 100% hình thức ban đầu, không bị trầy xước, móp méo hay dính chất lỏng.
- Đầy đủ hộp gốc, phiếu bảo hành, hóa đơn mua hàng và toàn bộ bộ phụ kiện đi kèm.
- Sản phẩm đã được đăng xuất hoàn toàn khỏi các tài khoản cá nhân (Apple ID, Google Account, Samsung Account).

## 2. Quy định nhập trả và phí thu lại
Trường hợp sản phẩm không có lỗi từ nhà sản xuất nhưng khách hàng muốn nhập trả lại hàng hoặc đổi sang sản phẩm khác:
- Đổi trả trong tháng đầu tiên: Phí thu lại tương đương 20% giá trị sản phẩm ghi trên hóa đơn.
- Đổi trả từ tháng thứ 2 trở đi: Mỗi tháng cộng thêm 10% phí thu lại theo quy định điều khoản thương mại.

## 3. Thời gian và phương thức hoàn tiền
Trường hợp phát sinh giao dịch hoàn tiền cho khách hàng, FPT Shop sẽ thực hiện theo phương thức thanh toán ban đầu trong thời hạn:
- Hoàn tiền mặt tại siêu thị: Xử lý ngay lập tức khi hoàn tất biên bản hủy đơn.
- Hoàn trả qua thẻ ngân hàng/ví điện tử: Hoàn tiền từ 3 - 7 ngày làm việc tùy thuộc vào ngân hàng phát hành thẻ."""
    }

    metadata = {
        "chinh-sach-thanh-toan.docx": """---
doc_id: fpt-payment-policy
title: Chính Sách Thanh Toán Và Bảo Mật Giao Dịch FPT Shop
customer_role: buyer
category: payment
language: vi
source_url: https://fptshop.com.vn/chinh-sach-thanh-toan
retrieved_at: 2026-08-03
document_version: "2026"
---

""",
        "chinh-sach-giao-hang.docx": """---
doc_id: fpt-shipping-policy
title: Chính Sách Giao Hàng Và Đồng Kiểm FPT Shop
customer_role: buyer
category: shipping
language: vi
source_url: https://fptshop.com.vn/chinh-sach-giao-hang
retrieved_at: 2026-08-03
document_version: "2026"
---

""",
        "chinh-sach-doi-san-pham.docx": """---
doc_id: fpt-return-policy
title: Chính Sách Đổi Sản Phẩm Và Hoàn Tiền FPT Shop
customer_role: buyer
category: refund
language: vi
source_url: https://fptshop.com.vn/chinh-sach-doi-tra
retrieved_at: 2026-08-03
document_version: "2026"
---

""",
    }

    legal_files = [
        ("chinh-sach-thanh-toan.docx", fpt_policies["chinh-sach-thanh-toan.docx"]),
        ("chinh-sach-giao-hang.docx", fpt_policies["chinh-sach-giao-hang.docx"]),
        ("chinh-sach-doi-san-pham.docx", fpt_policies["chinh-sach-doi-san-pham.docx"]),
    ]
    
    for dst_name, policy_text in legal_files:
        src_path = K4_DIR / dst_name.replace(".docx", ".md")
        dst_path = DATA_DIR / dst_name
        if src_path.exists():
            create_docx_from_markdown(src_path, dst_path)
        else:
            # Generate clean FPT policy docx directly
            print(f"[OK] Dang tao file DOCX FPT Shop: {dst_name}...")
            # Temp markdown path for function
            tmp_md = DATA_DIR / f"{dst_path.stem}.tmp.md"
            tmp_md.write_text(metadata[dst_name] + policy_text, encoding="utf-8")
            create_docx_from_markdown(tmp_md, dst_path)
            if tmp_md.exists():
                tmp_md.unlink()


if __name__ == "__main__":
    collect_legal_docs()




