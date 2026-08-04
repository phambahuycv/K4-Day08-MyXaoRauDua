"""
RAG Chatbot — E-commerce Support
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS — Premium Dark Theme
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styling */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.main-header h1 {
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown label {
    color: #b0b0d0;
}

/* Suggestion buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    color: #c0c0ff;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    text-align: left;
    transition: all 0.3s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.35) 0%, rgba(118, 75, 162, 0.35) 100%);
    border-color: rgba(102, 126, 234, 0.6);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
}

/* Source card styling */
.source-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.2s ease;
}
.source-card:hover {
    border-color: rgba(102, 126, 234, 0.4);
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.source-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #667eea;
}
.source-score {
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
}
.score-high { background: rgba(46, 213, 115, 0.15); color: #2ed573; }
.score-medium { background: rgba(255, 165, 2, 0.15); color: #ffa502; }
.score-low { background: rgba(255, 71, 87, 0.15); color: #ff4757; }
.source-content {
    font-size: 0.85rem;
    color: #8888aa;
    line-height: 1.5;
}
.source-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.3rem;
}
.badge-legal { background: rgba(52, 152, 219, 0.15); color: #3498db; }
.badge-news { background: rgba(155, 89, 182, 0.15); color: #9b59b6; }
.badge-hybrid { background: rgba(46, 213, 115, 0.15); color: #2ed573; }
.badge-pageindex { background: rgba(241, 196, 15, 0.15); color: #f1c40f; }

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #667eea;
}
.stat-label {
    font-size: 0.75rem;
    color: #8888aa;
    margin-top: 0.2rem;
}

/* Pipeline badge */
.pipeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
.pipeline-hybrid {
    background: rgba(46, 213, 115, 0.12);
    color: #2ed573;
    border: 1px solid rgba(46, 213, 115, 0.25);
}
.pipeline-pageindex {
    background: rgba(241, 196, 15, 0.12);
    color: #f1c40f;
    border: 1px solid rgba(241, 196, 15, 0.25);
}

/* Architecture diagram */
.arch-flow {
    background: rgba(102, 126, 234, 0.06);
    border: 1px solid rgba(102, 126, 234, 0.15);
    border-radius: 10px;
    padding: 0.8rem;
    font-size: 0.78rem;
    color: #9090b0;
    text-align: center;
    line-height: 1.8;
}
.arch-flow .step {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    background: rgba(102, 126, 234, 0.12);
    border-radius: 6px;
    color: #a0a0ff;
    font-weight: 500;
}
.arch-flow .arrow {
    color: #667eea;
    font-weight: bold;
    margin: 0 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.markdown("### 🛒 E-commerce RAG")
    st.caption("Trợ lý AI hỏi đáp chính sách thương mại điện tử — hỗ trợ khách hàng về đổi trả, thanh toán, giao hàng, bảo mật, quy định người bán.")

    st.divider()

    st.markdown("#### 💡 Câu hỏi gợi ý")
    suggestions = [
        "Chính sách đổi trả sản phẩm tại FPT Shop như thế nào?",
        "FPT Shop hỗ trợ những phương thức thanh toán nào?",
        "Thời gian giao hàng nội thành HN và HCM?",
        "Người mua TikTok Shop được hoàn hàng trong mấy ngày?",
        "Quy định về vi phạm sở hữu trí tuệ trên TikTok Shop?",
        "Bao nhiêu điểm vi phạm thì bị hủy gian hàng TikTok?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.markdown("#### ⚙️ Cấu hình Pipeline")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5, help="Số lượng đoạn văn bản sẽ được truy xuất từ vector store")

    st.divider()
    st.markdown("#### 🏗️ Kiến trúc hệ thống")
    st.markdown("""
    <div class="arch-flow">
        <span class="step">Query</span> <span class="arrow">→</span>
        <span class="step">Semantic + BM25</span> <span class="arrow">→</span>
        <span class="step">RRF Rerank</span> <span class="arrow">→</span>
        <span class="step">Fallback?</span> <span class="arrow">→</span>
        <span class="step">LLM + Citation</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("📦 Stack: ChromaDB · BAAI/bge-m3 · OpenRouter · Streamlit")


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🛒 E-commerce Support RAG Chatbot</h1>
    <p>Hệ thống hỏi đáp thông minh về chính sách thương mại điện tử — Hybrid Retrieval + LLM Generation có trích dẫn nguồn</p>
</div>
""", unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            retrieval_src = msg.get("retrieval_source", "hybrid")
            pipeline_class = "pipeline-hybrid" if retrieval_src == "hybrid" else "pipeline-pageindex"
            pipeline_emoji = "🔀" if retrieval_src == "hybrid" else "📄"
            st.markdown(
                f'<div class="pipeline-badge {pipeline_class}">'
                f'{pipeline_emoji} Retrieval: {retrieval_src.upper()}</div>',
                unsafe_allow_html=True,
            )

            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("title") or meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    score_class = "score-high" if score >= 0.7 else ("score-medium" if score >= 0.4 else "score-low")
                    badge_class = "badge-legal" if doc_type == "legal" else "badge-news"
                    content_preview = src.get("content", "")[:250]

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-name">📄 [{i}] {source_name}</span>
                            <span class="source-score {score_class}">⚡ {score:.4f}</span>
                        </div>
                        <div>
                            <span class="source-badge {badge_class}">{doc_type}</span>
                            <span class="source-badge badge-hybrid">{src.get("source", "hybrid")}</span>
                        </div>
                        <div class="source-content" style="margin-top:0.5rem">{content_preview}...</div>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("💬 Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        start_time = time.time()

        with st.spinner("🔍 Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                from src.task10_generation import generate_with_citation
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "hybrid")

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
                retrieval_source = "none"
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []
                retrieval_source = "error"

        elapsed = time.time() - start_time
        st.session_state.total_queries += 1

        st.markdown(answer)

        # Pipeline info badge
        if retrieval_source not in ("none", "error"):
            pipeline_class = "pipeline-hybrid" if retrieval_source == "hybrid" else "pipeline-pageindex"
            pipeline_emoji = "🔀" if retrieval_source == "hybrid" else "📄"
            st.markdown(
                f'<div class="pipeline-badge {pipeline_class}">'
                f'{pipeline_emoji} Retrieval: {retrieval_source.upper()} · ⏱️ {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

        # Sources section
        if sources:
            with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=True):
                for i, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("title") or meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    score_class = "score-high" if score >= 0.7 else ("score-medium" if score >= 0.4 else "score-low")
                    badge_class = "badge-legal" if doc_type == "legal" else "badge-news"
                    content_preview = src.get("content", "")[:250]

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-name">📄 [{i}] {source_name}</span>
                            <span class="source-score {score_class}">⚡ {score:.4f}</span>
                        </div>
                        <div>
                            <span class="source-badge {badge_class}">{doc_type}</span>
                            <span class="source-badge badge-hybrid">{src.get("source", "hybrid")}</span>
                        </div>
                        <div class="source-content" style="margin-top:0.5rem">{content_preview}...</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
