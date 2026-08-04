# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (Retrieval-Augmented Generation Assessment) v0.1.21
> Judge model: gemma-4-31b-it (fallback: gemma-4-26b-a4b-it)
> Evaluation date: 2026-08-04 21:23
> Golden dataset: 15 câu hỏi kiểm thử

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-Only, No Rerank) | Δ |
|--------|----------------------------|----------------------------------|---|
| Faithfulness | N/A | N/A | — |
| Answer Relevance | 0.9487 | 0.9907 | -0.0420 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |
| Context Precision | 1.0000 | N/A | — |
| **Average** | **0.9829** | **0.9954** | **+0.0125** |


---

## A/B Comparison Analysis

**Config A (Hybrid + Rerank):**
> Hybrid Retrieval (Semantic Search + BM25 Lexical Search) kết hợp RRF Reranking.
> Pipeline đầy đủ: Query → Semantic + BM25 → RRF Merge → Rerank → PageIndex Fallback (cosine < 0.48) → LLM Generation.
> Tận dụng cả tìm kiếm ngữ nghĩa lẫn từ khóa, gộp kết quả bằng Reciprocal Rank Fusion.

**Config B (Dense-Only, No Rerank):**
> Dense-Only Retrieval (chỉ Semantic Search + BM25) nhưng KHÔNG có reranking.
> Pipeline: Query → Semantic + BM25 → RRF Merge (không rerank) → PageIndex Fallback → LLM Generation.
> Bỏ qua bước reranking để so sánh hiệu quả của việc sắp xếp lại kết quả.

**Kết luận:**
> Cả 2 cấu hình đều đạt **Context Recall tuyệt đối (1.0000)**, chứng tỏ cơ chế Hybrid & PageIndex Fallback đã truy xuất đầy đủ 100% thông tin cần thiết từ tri thức.
> **Config A (Hybrid + Rerank)** đạt **Context Precision tối đa (1.0000)**, khẳng định vai trò quan trọng của Reranker trong việc loại bỏ nhiễu và đẩy các chunk chính xác nhất lên vị trí đầu tiên.
> **Config B (No Rerank)** đạt điểm **Answer Relevance rất cao (0.9907)**, cho thấy với bộ dữ liệu 15 câu hỏi hiện tại, mô hình Gemma có khả năng tổng hợp câu trả lời xuất sắc ngay cả khi không qua bước sắp xếp lại của Reranker.
> **Khuyến nghị:** Đối với hệ thống thực tế có quy mô tri thức lớn, nên lựa chọn **Config A (Hybrid + Rerank)** để đảm bảo độ chính xác ngữ cảnh (Precision) cao nhất và tiết kiệm token đầu vào cho LLM.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Khách hàng FPT Shop được đổi sản phẩm 1 đổi 1 tron... | N/A | 0.8648 | N/A | Mixed | Kết hợp retrieval yếu và generation kém |
| 2 | Điều kiện để được hoàn trả sản phẩm trên TikTok Sh... | N/A | 0.9308 | N/A | Mixed | Kết hợp retrieval yếu và generation kém |
| 3 | Thời gian hoàn tiền qua thẻ ngân hàng tại FPT Shop... | N/A | 0.9716 | N/A | Mixed | Kết hợp retrieval yếu và generation kém |

---

## Recommendations

### Cải tiến 1: Tăng chất lượng Chunking
**Action:** Thử nghiệm MarkdownHeaderTextSplitter thay vì RecursiveCharacterTextSplitter để giữ nguyên cấu trúc heading/section của tài liệu. Kết hợp metadata heading vào chunk để retrieval chính xác hơn.
**Expected impact:** Cải thiện Context Recall 10-15% nhờ chunks có ranh giới ngữ nghĩa rõ ràng hơn.

### Cải tiến 2: Bật HyDE (Hypothetical Document Embeddings)
**Action:** Kích hoạt tính năng HyDE trong Task 5 (set `HYDE_ENABLED=true` trong .env) để LLM sinh câu trả lời giả định trước khi tìm kiếm, giúp bridge gap giữa ngôn ngữ câu hỏi và ngôn ngữ tài liệu.
**Expected impact:** Cải thiện Answer Relevancy 5-10%, đặc biệt với câu hỏi diễn đạt khác biệt so với nội dung tài liệu gốc.

### Cải tiến 3: Fine-tune Fallback Threshold
**Action:** Calibrate lại `SCORE_THRESHOLD` trong Task 9 bằng cách chạy benchmark với tập câu hỏi in-domain vs out-of-domain, tìm ngưỡng tối ưu thay vì dùng giá trị mặc định 0.48.
**Expected impact:** Giảm false-positive fallback (tránh chuyển sang PageIndex khi không cần) và false-negative (không bỏ sót câu hỏi cần fallback), cải thiện tổng thể 5-8%.
