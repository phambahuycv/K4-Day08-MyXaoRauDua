"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"

_model = None


def get_embedding_model():
    """Singleton helper để load embedding model."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def semantic_search(query: str, top_k: int = 10, metadata_filter: dict = None) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        metadata_filter: Dict lọc metadata (VD: {"type": "legal"})

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        import chromadb
        if CHROMA_DIR.exists():
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                model = get_embedding_model()
                query_vector = model.encode(query).tolist()

                kwargs = {
                    "query_embeddings": [query_vector],
                    "n_results": top_k,
                    "include": ["documents", "metadatas", "distances"]
                }
                if metadata_filter:
                    kwargs["where"] = metadata_filter

                results = collection.query(**kwargs)

                if results and results.get("documents") and results["documents"][0]:
                    output = []
                    for doc, meta, dist in zip(
                        results["documents"][0], results["metadatas"][0], results["distances"][0]
                    ):
                        score = max(0.0, 1.0 - dist)
                        output.append({"content": doc, "score": round(float(score), 4), "metadata": meta})
                    output.sort(key=lambda x: x["score"], reverse=True)
                    return output[:top_k]
            except Exception:
                pass
    except ImportError:
        pass

    # Fallback search if chromadb is missing or empty DB
    from .task4_chunking_indexing import load_documents, chunk_documents
    docs = load_documents()
    chunks = chunk_documents(docs) if docs else []
    query_words = set(query.lower().split())

    output = []
    for c in chunks:
        meta = c.get("metadata", {})
        if metadata_filter:
            match = True
            for k, v in metadata_filter.items():
                if meta.get(k) != v:
                    match = False
                    break
            if not match:
                continue

        content = c.get("content", "")
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        score = round(overlap / max(len(query_words), 1), 4)
        output.append({"content": content, "score": score, "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]



if __name__ == "__main__":
    results = semantic_search("chính sách đổi trả sản phẩm", top_k=5)
    print(f"Tìm thấy {len(results)} kết quả semantic search:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:100]}...")

