"""
RAG Evaluation Pipeline.

Sử dụng RAGAS để đánh giá chất lượng RAG pipeline.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Chạy:
    python -m group_project.evaluation.eval_pipeline

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
CHECKPOINT_PATH = Path(__file__).parent / "eval_checkpoint.json"
RAGAS_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RAGAS_GEMINI_MODEL = os.getenv("RAGAS_GEMINI_MODEL") or "gemma-4-31b-it"
RAGAS_GEMINI_FALLBACK_MODEL = (
    os.getenv("RAGAS_GEMINI_FALLBACK_MODEL") or "gemma-4-26b-a4b-it"
)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_checkpoint(data: dict):
    try:
        content = json.dumps(data, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"  ⚠️ Could not save checkpoint: {e}")


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG Pipeline Wrapper
# =============================================================================

class RAGPipeline:
    """Wrapper quanh generate_with_citation để tiện truyền vào RAGAS."""

    def __init__(self, use_reranking: bool = True, top_k: int = 5):
        self.use_reranking = use_reranking
        self.top_k = top_k

    def generate_with_citation(self, query: str) -> dict:
        from src.task10_generation import (
            generate_with_citation,
            reorder_for_llm,
            format_context,
            SYSTEM_PROMPT,
            TEMPERATURE,
            TOP_P,
            LLM_MODEL,
        )
        from src.task9_retrieval_pipeline import retrieve

        chunks = retrieve(
            query,
            top_k=self.top_k,
            use_reranking=self.use_reranking,
        )

        if not chunks:
            return {
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "sources": [],
                "retrieval_source": "none",
            }

        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        gemini_key = os.getenv("GEMINI_API_KEY")
        from openai import OpenAI

        answer = None
        if gemini_key:
            try:
                client = OpenAI(
                    api_key=gemini_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=2,
                    timeout=30,
                )
                response = client.chat.completions.create(
                    model="gemma-4-31b-it",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"    ⚠️ Gemini API gemma-4-31b-it error ({e}), trying gemma-4-26b-a4b-it...")
                try:
                    response = client.chat.completions.create(
                        model="gemma-4-26b-a4b-it",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                    answer = response.choices[0].message.content
                except Exception as e2:
                    print(f"    ⚠️ Gemini API gemma-4-26b-a4b-it error ({e2})")

        if not answer:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        max_retries=1,
                        timeout=25,
                        default_headers={
                            "HTTP-Referer": "https://github.com/rag-lab",
                            "X-Title": "E-commerce Support RAG",
                        },
                    )
                    response = client.chat.completions.create(
                        model="google/gemma-4-31b-it:free",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    print(f"    ⚠️ OpenRouter error ({e})")

        return {
            "answer": answer or "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def _build_ragas_llm(model: str):
    """Tạo judge LLM sử dụng model gemma trên Gemini API (hoặc OpenRouter nếu cần)."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and ("gemma" in model or "gemini" in model):
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=RAGAS_GEMINI_BASE_URL,
            temperature=0,
            n=1,
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=model,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            n=1,
        )

    if api_key:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=RAGAS_GEMINI_BASE_URL,
            temperature=0,
            n=1,
        )

    raise RuntimeError("Cần cấu hình GEMINI_API_KEY hoặc OPENROUTER_API_KEY trong .env")


def _is_model_unavailable(error: Exception) -> bool:
    """Chỉ fallback khi Gemini báo model không tồn tại/không được hỗ trợ."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("model not found", "not_found", "not supported", "404")
    )


def collect_rag_responses(
    pipeline: RAGPipeline,
    golden_dataset: list[dict],
    cache_key: str = "config",
    checkpoint: dict = None,
) -> dict:
    """Thu thập responses từ RAG pipeline cho evaluation (có checkpoint từng câu hỏi)."""
    if checkpoint is None:
        checkpoint = {}

    cached_list = checkpoint.get(cache_key, [])
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(golden_dataset):
        if i < len(cached_list) and "Error:" not in cached_list[i].get("answer", "Error:"):
            print(f"  ⏭️ [{i+1}/{len(golden_dataset)}] Skipped (loaded from cache): {item['question'][:40]}...")
            res = cached_list[i]
            eval_data["question"].append(res["question"])
            eval_data["answer"].append(res["answer"])
            eval_data["contexts"].append(res["contexts"])
            eval_data["ground_truth"].append(res["ground_truth"])
            continue

        print(f"  📝 [{i+1}/{len(golden_dataset)}] {item['question'][:60]}...")
        try:
            result = pipeline.generate_with_citation(item["question"])
            ans = result.get("answer", "")
            ctxs = [chunk.get("content", "") for chunk in result.get("sources", [])]
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            ans = f"Error: {e}"
            ctxs = ["Không có context do lỗi"]

        record = {
            "question": item["question"],
            "answer": ans,
            "contexts": ctxs,
            "ground_truth": item["expected_answer"],
        }
        eval_data["question"].append(record["question"])
        eval_data["answer"].append(record["answer"])
        eval_data["contexts"].append(record["contexts"])
        eval_data["ground_truth"].append(record["ground_truth"])

        if i < len(cached_list):
            cached_list[i] = record
        else:
            cached_list.append(record)

        checkpoint[cache_key] = cached_list
        save_checkpoint(checkpoint)
        time.sleep(1)

    return eval_data


def _build_ragas_embeddings():
    """Tạo embeddings cho RAGAS sử dụng HuggingFace BAAI/bge-m3 (local) hoặc Gemini API."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY chưa được cấu hình trong .env")

        return OpenAIEmbeddings(
            model="text-embedding-004",
            api_key=api_key,
            base_url=RAGAS_GEMINI_BASE_URL,
        )


def evaluate_with_ragas(eval_data: dict) -> "pandas.DataFrame":
    """
    Evaluate sử dụng RAGAS framework.

    Returns:
        pandas DataFrame with per-question scores.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    answer_relevancy.strictness = 1

    dataset = Dataset.from_dict(eval_data)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    embeddings = _build_ragas_embeddings()

    from ragas.run_config import RunConfig

    models = []
    for m in ["gemma-4-26b-a4b-it", "gemma-4-31b-it", RAGAS_GEMINI_MODEL, RAGAS_GEMINI_FALLBACK_MODEL]:
        if m and m not in models:
            models.append(m)

    last_error = None
    for index, model in enumerate(models):
        try:
            print(f"  🤖 Trying RAGAS judge model: {model} (max_workers=8)")
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=_build_ragas_llm(model),
                embeddings=embeddings,
                run_config=RunConfig(max_workers=8, max_retries=5, max_wait=120),
            )
            df = result.to_pandas()
            if df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].isna().all().all():
                print(f"  ⚠️ Model {model} trả về toàn NaN (do hết quota/rate limit), thử model tiếp theo...")
                continue
            return df
        except Exception as error:
            last_error = error
            is_last_model = index == len(models) - 1
            if is_last_model:
                print(f"  ❌ Đã thử tất cả model nhưng thất bại: {error}")
                raise
            print(f"  ⚠️ Lỗi với model {model} ({error}), chuyển sang model tiếp theo...")

    raise RuntimeError(
        "Không thể chạy RAGAS với các model Gemini đã cấu hình"
    ) from last_error


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - Config A: hybrid search + reranking (full pipeline)
    - Config B: dense-only (không reranking)
    """
    import pandas as pd

    configs = {
        "Config A (Hybrid + Rerank)": RAGPipeline(use_reranking=True, top_k=5),
        "Config B (Dense-Only, No Rerank)": RAGPipeline(use_reranking=False, top_k=5),
    }

    checkpoint = load_checkpoint()
    all_results = {}

    for config_name, pipeline in configs.items():
        print(f"\n{'='*60}")
        print(f"🔬 Running: {config_name}")
        print(f"{'='*60}")

        # Collect responses (với checkpointing)
        print("📥 Collecting RAG responses...")
        eval_data = collect_rag_responses(
            pipeline,
            golden_dataset,
            cache_key=f"{config_name}_responses",
            checkpoint=checkpoint,
        )

        # Run RAGAS evaluation (hoặc load từ cache)
        print("📊 Running RAGAS evaluation...")
        cached_scores = checkpoint.get(f"{config_name}_scores")
        cached_df_list = checkpoint.get(f"{config_name}_df")

        if cached_scores and cached_df_list:
            print(f"  ⏭️ [Cache Hit] Loaded RAGAS scores from eval_checkpoint.json for {config_name}!")
            df = pd.DataFrame(cached_df_list)
            all_results[config_name] = {
                "scores": cached_scores,
                "per_question": df,
                "error": None,
            }
            continue

        try:
            df = evaluate_with_ragas(eval_data)
            metrics_avg = {
                "faithfulness": df["faithfulness"].mean() if "faithfulness" in df else 0,
                "answer_relevancy": df["answer_relevancy"].mean() if "answer_relevancy" in df else 0,
                "context_recall": df["context_recall"].mean() if "context_recall" in df else 0,
                "context_precision": df["context_precision"].mean() if "context_precision" in df else 0,
            }
            all_results[config_name] = {
                "scores": metrics_avg,
                "per_question": df,
                "error": None,
            }
            # Lưu ngay vào checkpoint
            checkpoint[f"{config_name}_scores"] = metrics_avg
            checkpoint[f"{config_name}_df"] = df.to_dict(orient="records")
            save_checkpoint(checkpoint)
            print(f"  ✅ Done! Average scores: {metrics_avg}")
        except Exception as e:
            print(f"  ❌ RAGAS evaluation failed: {e}")
            traceback.print_exc()
            all_results[config_name] = {
                "scores": {
                    "faithfulness": 0,
                    "answer_relevancy": 0,
                    "context_recall": 0,
                    "context_precision": 0,
                },
                "per_question": None,
                "error": str(e),
            }

    return all_results


# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict, golden_dataset: list[dict]):
    """Export evaluation results to results.md"""

    config_names = list(comparison.keys())
    config_a_name = config_names[0] if len(config_names) > 0 else "Config A"
    config_b_name = config_names[1] if len(config_names) > 1 else "Config B"

    scores_a = comparison.get(config_a_name, {}).get("scores", {})
    scores_b = comparison.get(config_b_name, {}).get("scores", {})
    error_a = comparison.get(config_a_name, {}).get("error")
    error_b = comparison.get(config_b_name, {}).get("error")
    df_a = comparison.get(config_a_name, {}).get("per_question")
    df_b = comparison.get(config_b_name, {}).get("per_question")

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) and val > 0 else "N/A"

    def delta(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
            diff = a - b
            return f"{diff:+.4f}"
        return "—"

    # Calculate averages
    avg_a = sum(v for v in scores_a.values() if isinstance(v, (int, float))) / max(len(scores_a), 1)
    avg_b = sum(v for v in scores_b.values() if isinstance(v, (int, float))) / max(len(scores_b), 1)

    content = f"""# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (Retrieval-Augmented Generation Assessment) v0.1.21
> Judge model: {RAGAS_GEMINI_MODEL} (fallback: {RAGAS_GEMINI_FALLBACK_MODEL})
> Evaluation date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
> Golden dataset: {len(golden_dataset)} câu hỏi kiểm thử

---

## Overall Scores

| Metric | {config_a_name} | {config_b_name} | Δ |
|--------|{'-' * len(config_a_name)}--|{'-' * len(config_b_name)}--|---|
| Faithfulness | {fmt(scores_a.get("faithfulness", 0))} | {fmt(scores_b.get("faithfulness", 0))} | {delta(scores_a.get("faithfulness", 0), scores_b.get("faithfulness", 0))} |
| Answer Relevance | {fmt(scores_a.get("answer_relevancy", 0))} | {fmt(scores_b.get("answer_relevancy", 0))} | {delta(scores_a.get("answer_relevancy", 0), scores_b.get("answer_relevancy", 0))} |
| Context Recall | {fmt(scores_a.get("context_recall", 0))} | {fmt(scores_b.get("context_recall", 0))} | {delta(scores_a.get("context_recall", 0), scores_b.get("context_recall", 0))} |
| Context Precision | {fmt(scores_a.get("context_precision", 0))} | {fmt(scores_b.get("context_precision", 0))} | {delta(scores_a.get("context_precision", 0), scores_b.get("context_precision", 0))} |
| **Average** | **{fmt(avg_a)}** | **{fmt(avg_b)}** | **{delta(avg_a, avg_b)}** |

"""

    if error_a:
        content += f"\n> ⚠️ {config_a_name} gặp lỗi: {error_a}\n"
    if error_b:
        content += f"\n> ⚠️ {config_b_name} gặp lỗi: {error_b}\n"

    content += f"""
---

## A/B Comparison Analysis

**{config_a_name}:**
> Hybrid Retrieval (Semantic Search + BM25 Lexical Search) kết hợp RRF Reranking.
> Pipeline đầy đủ: Query → Semantic + BM25 → RRF Merge → Rerank → PageIndex Fallback (cosine < 0.48) → LLM Generation.
> Tận dụng cả tìm kiếm ngữ nghĩa lẫn từ khóa, gộp kết quả bằng Reciprocal Rank Fusion.

**{config_b_name}:**
> Dense-Only Retrieval (chỉ Semantic Search + BM25) nhưng KHÔNG có reranking.
> Pipeline: Query → Semantic + BM25 → RRF Merge (không rerank) → PageIndex Fallback → LLM Generation.
> Bỏ qua bước reranking để so sánh hiệu quả của việc sắp xếp lại kết quả.

**Kết luận:**
"""

    if avg_a > avg_b and avg_a > 0:
        content += f"""> {config_a_name} cho kết quả tốt hơn với điểm trung bình {fmt(avg_a)} so với {fmt(avg_b)} của {config_b_name}.
> Reranking giúp cải thiện chất lượng kết quả trả về bằng cách sắp xếp lại các chunks theo độ liên quan với câu hỏi.
"""
    elif avg_b > avg_a and avg_b > 0:
        content += f"""> {config_b_name} cho kết quả tốt hơn với điểm trung bình {fmt(avg_b)} so với {fmt(avg_a)} của {config_a_name}.
> Trong trường hợp này, reranking không mang lại cải thiện đáng kể, có thể do corpus nhỏ và RRF đã đủ hiệu quả.
"""
    else:
        content += "> Chưa có đủ dữ liệu đánh giá để kết luận. Cần chạy lại evaluation với GEMINI_API_KEY hợp lệ.\n"

    content += """
---

## Worst Performers (Bottom 3)

"""

    # Try to find worst performers from Config A
    if df_a is not None and not df_a.empty:
        metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_recall"] if c in df_a.columns]
        if metric_cols:
            df_a["avg_score"] = df_a[metric_cols].mean(axis=1)
            worst = df_a.nsmallest(min(3, len(df_a)), "avg_score")

            content += f"| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
            content += f"|---|----------|-------------|-----------|--------|---------------|------------|\n"

            for idx, (_, row) in enumerate(worst.iterrows(), 1):
                q = row.get("question", "N/A")[:50] + "..."
                faith = fmt(row.get("faithfulness", 0))
                rel = fmt(row.get("answer_relevancy", 0))
                rec = fmt(row.get("context_recall", 0))
                # Determine failure stage
                if row.get("context_recall", 1) < 0.5:
                    stage = "Retrieval"
                    cause = "Context không chứa đủ thông tin cần thiết"
                elif row.get("faithfulness", 1) < 0.5:
                    stage = "Generation"
                    cause = "LLM bịa thêm thông tin ngoài context"
                else:
                    stage = "Mixed"
                    cause = "Kết hợp retrieval yếu và generation kém"
                content += f"| {idx} | {q} | {faith} | {rel} | {rec} | {stage} | {cause} |\n"
        else:
            content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
            content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"
            content += "| — | Chưa có dữ liệu | — | — | — | — | Chạy lại eval_pipeline |\n"
    else:
        content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
        content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"
        content += "| — | Chưa có dữ liệu | — | — | — | — | Cần GEMINI_API_KEY hợp lệ để chạy RAGAS |\n"

    content += """
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
"""

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n📄 Results exported to: {RESULTS_PATH}")
    return content


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 RAG Evaluation Pipeline — RAGAS Framework")
    print("=" * 60)

    if "--clear-cache" in sys.argv and CHECKPOINT_PATH.exists():
        try:
            CHECKPOINT_PATH.unlink()
            print("🗑️ Đã xóa cache cũ (eval_checkpoint.json).")
        except Exception as e:
            print(f"⚠️ Không thể xóa cache: {e}")

    # Load golden dataset
    golden_dataset = load_golden_dataset()
    print(f"\n📂 Loaded {len(golden_dataset)} test cases from golden_dataset.json")

    # Check for required API keys
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

    if not has_openrouter:
        print("\n❌ OPENROUTER_API_KEY chưa được cấu hình trong .env")
        print("   Cần API key để gọi LLM sinh câu trả lời.")
        sys.exit(1)

    if not has_gemini:
        print("\n⚠️  GEMINI_API_KEY chưa được cấu hình trong .env")
        print("   Sẽ chạy thu thập responses nhưng KHÔNG thể chạy RAGAS evaluation.")
        print("   Để chạy RAGAS, thêm GEMINI_API_KEY vào .env")
        print("   (Lấy miễn phí tại https://aistudio.google.com/apikey)")
        print()

        # Still collect responses and export a partial results
        print("📥 Thu thập responses với Config A (Hybrid + Rerank)...")
        pipeline = RAGPipeline(use_reranking=True, top_k=5)
        eval_data_a = collect_rag_responses(pipeline, golden_dataset)

        print("\n📥 Thu thập responses với Config B (Dense-Only, No Rerank)...")
        pipeline_b = RAGPipeline(use_reranking=False, top_k=5)
        eval_data_b = collect_rag_responses(pipeline_b, golden_dataset)

        # Export partial results without RAGAS scores
        comparison = {
            "Config A (Hybrid + Rerank)": {
                "scores": {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "context_precision": 0},
                "per_question": None,
                "error": "GEMINI_API_KEY not configured — RAGAS scoring skipped",
            },
            "Config B (Dense-Only, No Rerank)": {
                "scores": {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "context_precision": 0},
                "per_question": None,
                "error": "GEMINI_API_KEY not configured — RAGAS scoring skipped",
            },
        }
        export_results(comparison, golden_dataset)
        print("\n✅ Partial results exported (without RAGAS scores).")
        print("   Add GEMINI_API_KEY to .env and re-run for full evaluation.")
    else:
        # Full evaluation with RAGAS
        comparison = compare_configs(golden_dataset)
        export_results(comparison, golden_dataset)

    print("\n" + "=" * 60)
    print("✅ Evaluation pipeline completed!")
    print("=" * 60)
