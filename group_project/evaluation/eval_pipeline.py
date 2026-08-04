"""
RAG Evaluation Pipeline.

Pipeline này chạy đánh giá heuristic trực tiếp trên golden dataset và ghi kết quả
vào results.md để có thể xem nhanh 4 chỉ số: Faithfulness, Relevancy, Recall, Precision.
"""

import json
import re
from pathlib import Path

import src.task10_generation as task10_generation
from src.task9_retrieval_pipeline import retrieve as base_retrieve

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

STOPWORDS = {
    "của", "và", "các", "có", "là", "được", "trong", "tại", "theo", "với", "không",
    "của", "đối", "với", "sản", "phẩm", "câu", "hỏi", "những", "nếu", "khi", "một",
    "mà", "thì", "cũng", "cần", "cho", "đến", "từ", "ở", "vào", "như", "này", "đó"
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh token."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(text: str) -> set[str]:
    """Trích xuất từ khóa quan trọng từ text."""
    normalized = normalize_text(text)
    if not normalized:
        return set()
    tokens = [token for token in normalized.split() if len(token) > 1 and token not in STOPWORDS]
    return set(tokens)


def overlap_score(a_tokens: set[str], b_tokens: set[str]) -> float:
    """Tính điểm overlap bằng F1-style trên bag-of-words."""
    if not a_tokens and not b_tokens:
        return 0.0
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    precision = intersection / len(a_tokens)
    recall = intersection / len(b_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def build_pipeline(use_reranking: bool):
    """Tạo pipeline đánh giá với config use_reranking tương ứng."""
    def pipeline(query: str):
        task10_generation.retrieve = lambda q, top_k=5: base_retrieve(
            q,
            top_k=top_k,
            use_reranking=use_reranking,
        )
        return task10_generation.generate_with_citation(query, top_k=5)

    return pipeline


def evaluate_case(item: dict, pipeline) -> dict:
    """Đánh giá 1 test case bằng heuristic."""
    result = pipeline(item["question"])
    answer = result.get("answer", "") or ""
    sources = result.get("sources", []) or []

    answer_keywords = extract_keywords(answer)
    question_keywords = extract_keywords(item["question"])
    expected_context_keywords = extract_keywords(item.get("expected_context", ""))
    expected_answer_keywords = extract_keywords(item.get("expected_answer", ""))

    retrieved_context = "\n".join(
        chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        for chunk in sources
    )
    retrieved_keywords = extract_keywords(retrieved_context)

    if not answer or "không thể xác minh" in answer.lower():
        faithfulness = 0.0
    else:
        faithfulness = overlap_score(answer_keywords, retrieved_keywords)

    relevancy = overlap_score(question_keywords, answer_keywords)
    recall = overlap_score(expected_context_keywords, retrieved_keywords)

    if not sources:
        precision = 0.0
    else:
        relevant_chunks = 0
        for chunk in sources:
            chunk_text = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            chunk_keywords = extract_keywords(chunk_text)
            chunk_overlap = overlap_score(expected_context_keywords, chunk_keywords)
            if chunk_overlap >= 0.1 or overlap_score(expected_answer_keywords, chunk_keywords) >= 0.1:
                relevant_chunks += 1
        precision = round(relevant_chunks / len(sources), 4)

    return {
        "question": item["question"],
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "answer": answer,
    }


def evaluate_config(golden_dataset: list[dict], config_name: str, use_reranking: bool) -> dict:
    """Chạy toàn bộ golden dataset cho một config."""
    pipeline = build_pipeline(use_reranking=use_reranking)
    results = [evaluate_case(item, pipeline) for item in golden_dataset]

    def avg(metric: str) -> float:
        return round(sum(item[metric] for item in results) / len(results), 4) if results else 0.0

    return {
        "config_name": config_name,
        "results": results,
        "metrics": {
            "Faithfulness": avg("faithfulness"),
            "Relevancy": avg("relevancy"),
            "Recall": avg("recall"),
            "Precision": avg("precision"),
        },
        "average": round(
            (avg("faithfulness") + avg("relevancy") + avg("recall") + avg("precision")) / 4,
            4,
        ),
    }


def compare_configs(golden_dataset: list[dict]) -> dict:
    """So sánh 2 config: hybrid + rerank vs dense-only."""
    return {
        "hybrid_rerank": evaluate_config(golden_dataset, "hybrid_rerank", use_reranking=True),
        "dense_only": evaluate_config(golden_dataset, "dense_only", use_reranking=False),
    }


def export_results(comparison: dict):
    """Export evaluation results to results.md."""
    metrics = ["Faithfulness", "Relevancy", "Recall", "Precision"]
    config_a = comparison["hybrid_rerank"]
    config_b = comparison["dense_only"]

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    lines = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append("> Đánh giá heuristic trên golden dataset bằng pipeline retrieval + generation hiện có, không cần API bên ngoài.")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |")
    lines.append("|---|---:|---:|---:|")

    for metric in metrics:
        a_score = config_a["metrics"][metric]
        b_score = config_b["metrics"][metric]
        delta = round(a_score - b_score, 4)
        lines.append(f"| {metric} | {fmt(a_score)} | {fmt(b_score)} | {fmt(delta)} |")

    avg_a = config_a["average"]
    avg_b = config_b["average"]
    lines.append(f"| **Average** | **{fmt(avg_a)}** | **{fmt(avg_b)}** | **{fmt(avg_a - avg_b)}** |")
    lines.append("")
    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append("**Config A:** Hybrid search + reranking")
    lines.append("**Config B:** Dense-only retrieval without reranking")
    best_config = "Config A" if avg_a >= avg_b else "Config B"
    lines.append(f"**Kết luận:** {best_config} đạt điểm trung bình tốt hơn trong thử nghiệm hiện tại vì có thêm reranking giúp tăng độ liên quan và độ chính xác của context.")
    lines.append("")
    lines.append("## Worst Performers (Bottom 3)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevancy | Recall | Precision | Root Cause |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    worst_cases = sorted(
        config_a["results"],
        key=lambda item: (item["faithfulness"] + item["relevancy"] + item["recall"] + item["precision"]) / 4,
    )[:3]

    for idx, item in enumerate(worst_cases, 1):
        overall = (item["faithfulness"] + item["relevancy"] + item["recall"] + item["precision"]) / 4
        if item["recall"] < 0.2:
            root_cause = "Retriever thiếu evidence liên quan"
        elif item["faithfulness"] < 0.2:
            root_cause = "Answer không đủ căn cứ trong context"
        else:
            root_cause = "Context và answer chưa đủ phong phú"
        lines.append(
            f"| {idx} | {item['question']} | {fmt(item['faithfulness'])} | {fmt(item['relevancy'])} | {fmt(item['recall'])} | {fmt(item['precision'])} | {root_cause} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1")
    lines.append("**Action:** Tăng độ dài context và dùng reranking rõ ràng hơn cho các câu hỏi có từ khóa chuyên ngành.")
    lines.append("**Expected impact:** Tăng Recall và Precision cho các policy/query khó.")
    lines.append("")
    lines.append("### Cải tiến 2")
    lines.append("**Action:** Dùng prompt chặt chẽ hơn để bắt buộc câu trả lời chỉ dùng thông tin xuất hiện trong retrieved chunks.")
    lines.append("**Expected impact:** Tăng Faithfulness và giảm câu trả lời lan man.")
    lines.append("")
    lines.append("### Cải tiến 3")
    lines.append("**Action:** Cải thiện bộ tách chunk và mở rộng corpus bằng các tài liệu policy chi tiết hơn.")
    lines.append("**Expected impact:** Tăng độ bao phủ và độ liên quan cho các câu hỏi trung bình.")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print("Completed evaluation successfully")
