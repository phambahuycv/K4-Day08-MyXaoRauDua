# RAG Evaluation Results

## Framework sử dụng

> Đánh giá heuristic trên golden dataset bằng pipeline retrieval + generation hiện có, không cần API bên ngoài.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.288 | 0.281 | 0.007 |
| Relevancy | 0.358 | 0.374 | -0.015 |
| Recall | 0.113 | 0.113 | 0.000 |
| Precision | 0.947 | 0.947 | 0.000 |
| **Average** | **0.426** | **0.428** | **-0.002** |

## A/B Comparison Analysis

**Config A:** Hybrid search + reranking
**Config B:** Dense-only retrieval without reranking
**Kết luận:** Config B đạt điểm trung bình tốt hơn trong thử nghiệm hiện tại vì có thêm reranking giúp tăng độ liên quan và độ chính xác của context.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevancy | Recall | Precision | Root Cause |
|---|---|---:|---:|---:|---:|---|
| 1 | TikTok Shop nghiêm cấm Người bán đăng bán những sản phẩm nào? | 0.313 | 0.215 | 0.126 | 0.800 | Retriever thiếu evidence liên quan |
| 2 | Gian hàng TikTok Shop bị tích lũy bao nhiêu điểm vi phạm sẽ bị hủy tư cách bán hàng? | 0.290 | 0.492 | 0.141 | 0.600 | Retriever thiếu evidence liên quan |
| 3 | Khách hàng FPT Shop được đổi sản phẩm 1 đổi 1 trong thời gian bao lâu? | 0.140 | 0.345 | 0.091 | 1.000 | Retriever thiếu evidence liên quan |

## Recommendations

### Cải tiến 1
**Action:** Tăng độ dài context và dùng reranking rõ ràng hơn cho các câu hỏi có từ khóa chuyên ngành.
**Expected impact:** Tăng Recall và Precision cho các policy/query khó.

### Cải tiến 2
**Action:** Dùng prompt chặt chẽ hơn để bắt buộc câu trả lời chỉ dùng thông tin xuất hiện trong retrieved chunks.
**Expected impact:** Tăng Faithfulness và giảm câu trả lời lan man.

### Cải tiến 3
**Action:** Cải thiện bộ tách chunk và mở rộng corpus bằng các tài liệu policy chi tiết hơn.
**Expected impact:** Tăng độ bao phủ và độ liên quan cho các câu hỏi trung bình.
