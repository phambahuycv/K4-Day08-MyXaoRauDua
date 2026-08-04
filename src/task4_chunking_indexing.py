"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import ast
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from functools import lru_cache
from pathlib import Path

# sentence-transformers only needs PyTorch here. Prevent transformers from
# importing an unrelated TensorFlow installation that may conflict with protobuf.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive splitting giữ được ranh giới đoạn/tiêu đề tốt hơn cắt theo số byte.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        metadata = _parse_front_matter(content)
        metadata.update(
            {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": "legal" if md_file.parent.name == "legal" else "news",
            }
        )
        metadata.setdefault("customer_role", "both")
        metadata.setdefault("doc_id", md_file.stem)
        documents.append({"content": content, "metadata": metadata})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = lambda text: _fallback_split(text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks = []
    for doc in documents:
        for index, chunk_text in enumerate(split_text(doc["content"])):
            text = chunk_text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": index,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        [chunk["content"] for chunk in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        raise ValueError("Không có chunk để index")
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[
            f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
            for chunk in chunks
        ],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in chunks],
    )
    return collection


def _parse_front_matter(content: str) -> dict:
    """Đọc YAML front matter đơn giản trong các file Markdown của lab."""
    lines = content.splitlines()
    separators = [index for index, line in enumerate(lines) if line.strip() == "---"]
    for start, end in zip(separators, separators[1:]):
        metadata = {}
        for line in lines[start + 1 : end]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = value.strip('"\'')
            metadata[key.strip()] = str(value)
        if metadata and {"doc_id", "customer_role", "category"} & metadata.keys():
            return metadata
    return {}


def _chroma_metadata(metadata: dict) -> dict:
    """Chroma chỉ nhận metadata dạng scalar."""
    return {key: str(value) for key, value in metadata.items() if value is not None}


def _fallback_split(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)


def get_collection():
    import chromadb

    if not CHROMA_DIR.exists():
        raise FileNotFoundError("ChromaDB chưa tồn tại, hãy chạy Task 4 trước")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
