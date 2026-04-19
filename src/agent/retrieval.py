from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from src.agent.state import RetrievedChunk
from src.features import FEATURE_METADATA, derive_partial_risk_signals, format_feature_value
from src.utils import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR, humanize_slug

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as exc:  # pragma: no cover - exercised indirectly via status checks
    Chroma = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    Embeddings = Any  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    _RETRIEVAL_IMPORT_ERROR = exc
else:
    _RETRIEVAL_IMPORT_ERROR = None


CHROMA_COLLECTION_NAME = "cardio_guidelines"
CHROMA_PERSIST_DIR = VECTORSTORE_DIR / "chroma_db"
DEFAULT_TOP_K = 4


def _ensure_retrieval_dependencies() -> None:
    if _RETRIEVAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Vector retrieval dependencies are unavailable. Install `langgraph`, `langchain`, "
            "`langchain-chroma`, `langchain-text-splitters`, and `chromadb`."
        ) from _RETRIEVAL_IMPORT_ERROR


def get_vectorstore_path(persist_directory: str | Path | None = None) -> Path:
    """Resolve the persisted Chroma directory used by the app."""
    return Path(persist_directory) if persist_directory else CHROMA_PERSIST_DIR


def vectorstore_exists(persist_directory: str | Path | None = None) -> bool:
    """Return True when a persisted local vector store already exists."""
    base_dir = get_vectorstore_path(persist_directory)
    return base_dir.exists() and any(base_dir.iterdir())


def get_vectorstore_status(persist_directory: str | Path | None = None) -> dict[str, Any]:
    """Expose the current vector-store configuration for the UI and docs."""
    return {
        "backend": "chroma",
        "collection_name": CHROMA_COLLECTION_NAME,
        "persist_directory": str(get_vectorstore_path(persist_directory)),
        "persisted": vectorstore_exists(persist_directory),
        "dependencies_available": _RETRIEVAL_IMPORT_ERROR is None,
    }


def _parse_markdown_document(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    title = humanize_slug(path.stem)
    keywords: list[str] = []
    sections: list[dict[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    def flush_section() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append(
            {
                "section_heading": current_heading,
                "content": text,
            }
        )

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip() or title
            continue
        if stripped.lower().startswith("keywords:"):
            keywords = [item.strip().lower() for item in stripped.split(":", maxsplit=1)[1].split(",") if item.strip()]
            continue
        if stripped.startswith("## "):
            flush_section()
            current_heading = stripped[3:].strip() or current_heading
            current_lines = []
            continue
        current_lines.append(raw_line)

    flush_section()
    return {
        "source": path.name,
        "title": title,
        "content": content,
        "keywords": keywords,
        "sections": sections or [{"section_heading": "Overview", "content": content}],
    }


def load_knowledge_base(kb_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, Any]]:
    """Load all markdown guidance files from the local knowledge base."""
    base_dir = Path(kb_dir)
    return [_parse_markdown_document(path) for path in sorted(base_dir.glob("*.md"))]


def build_retrieval_query(
    patient_data: Mapping[str, Any] | None = None,
    question: str | None = None,
    risk_prediction: str | None = None,
    risk_factors: list[dict[str, Any]] | None = None,
) -> str:
    """Create a semantic retrieval query from the patient context and model output."""
    parts: list[str] = []

    if risk_prediction:
        parts.append(f"Predicted cardiovascular risk tier: {risk_prediction}.")

    if risk_factors:
        factor_labels = [str(item.get("feature", "")).strip() for item in risk_factors if item.get("feature")]
        if factor_labels:
            parts.append("Top risk factors: " + "; ".join(factor_labels[:4]) + ".")

    if patient_data:
        patient_context: list[str] = []
        for field, value in patient_data.items():
            if field not in FEATURE_METADATA:
                continue
            label = FEATURE_METADATA[field]["label"]
            patient_context.append(f"{label} = {format_feature_value(field, value)}")
        if patient_context:
            parts.append("Patient context: " + "; ".join(patient_context[:8]) + ".")

        signals = derive_partial_risk_signals(patient_data)
        if signals:
            parts.append("Clinical signals: " + "; ".join(signal["label"] for signal in signals[:5]) + ".")

    if question:
        parts.append(f"Clinical focus: {question.strip()}.")

    return " ".join(parts) or "General cardiovascular prevention, monitoring, diet, and warning signs guidance."


def _get_embeddings(embeddings: Embeddings | None = None) -> Embeddings:
    if embeddings is not None:
        return embeddings
    from src.agent.embeddings import LocalChromaEmbeddings

    return LocalChromaEmbeddings()


def _build_splitter() -> RecursiveCharacterTextSplitter:
    _ensure_retrieval_dependencies()
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def build_knowledge_documents(kb_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    """Chunk the markdown knowledge base into vector-store documents."""
    _ensure_retrieval_dependencies()
    splitter = _build_splitter()
    documents: list[Document] = []

    for source_document in load_knowledge_base(kb_dir=kb_dir):
        chunk_index = 1
        for section in source_document["sections"]:
            section_heading = section["section_heading"] or "Overview"
            for chunk in splitter.split_text(section["content"]):
                cleaned_chunk = chunk.strip()
                if not cleaned_chunk:
                    continue

                chunk_id = f"{Path(source_document['source']).stem}-{chunk_index:03d}"
                page_content = (
                    f"Document: {source_document['title']}\n"
                    f"Section: {section_heading}\n\n"
                    f"{cleaned_chunk}"
                )
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "chunk_id": chunk_id,
                            "source_file": source_document["source"],
                            "document_title": source_document["title"],
                            "section_heading": section_heading,
                            "raw_chunk": cleaned_chunk,
                        },
                    )
                )
                chunk_index += 1

    return documents


def load_vectorstore(
    persist_directory: str | Path | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load an existing persisted Chroma vector store."""
    _ensure_retrieval_dependencies()
    persist_dir = get_vectorstore_path(persist_directory)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=_get_embeddings(embeddings),
    )


def build_vectorstore(
    kb_dir: str | Path = KNOWLEDGE_BASE_DIR,
    persist_directory: str | Path | None = None,
    embeddings: Embeddings | None = None,
) -> dict[str, Any]:
    """Rebuild the local persistent Chroma store from the markdown knowledge base."""
    _ensure_retrieval_dependencies()
    persist_dir = get_vectorstore_path(persist_directory)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = build_knowledge_documents(kb_dir=kb_dir)
    if not documents:
        raise FileNotFoundError("No markdown guidance documents were found to index.")

    vectorstore = load_vectorstore(persist_directory=persist_dir, embeddings=embeddings)
    ids = [document.metadata["chunk_id"] for document in documents]
    vectorstore.add_documents(documents=documents, ids=ids)

    return {
        "backend": "chroma",
        "collection_name": CHROMA_COLLECTION_NAME,
        "persist_directory": str(persist_dir),
        "documents_indexed": len(documents),
        "source_files": sorted({document.metadata["source_file"] for document in documents}),
    }


def ensure_vectorstore(
    persist_directory: str | Path | None = None,
    kb_dir: str | Path = KNOWLEDGE_BASE_DIR,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load the persisted vector store, or build it automatically when missing."""
    persist_dir = get_vectorstore_path(persist_directory)
    if not vectorstore_exists(persist_dir):
        build_vectorstore(kb_dir=kb_dir, persist_directory=persist_dir, embeddings=embeddings)
    return load_vectorstore(persist_directory=persist_dir, embeddings=embeddings)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Cache the default vector store for repeated Streamlit workflow runs."""
    return ensure_vectorstore()


def format_retrieved_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Produce clean source labels for the UI and final report."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        section_heading = chunk["section_heading"] or "Overview"
        label = f"{chunk['document_title']} ({chunk['source_file']}, {section_heading}, {chunk['chunk_id']})"
        if label in seen:
            continue
        seen.add(label)
        sources.append(label)
    return sources


def retrieve_guideline_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: str | Path | None = None,
    embeddings: Embeddings | None = None,
) -> list[RetrievedChunk]:
    """Run semantic similarity search against the persisted vector store."""
    vectorstore = get_vectorstore() if persist_directory is None and embeddings is None else ensure_vectorstore(
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    chunks: list[RetrievedChunk] = []

    for document, score in results:
        metadata = document.metadata
        raw_chunk = metadata.get("raw_chunk") or document.page_content
        snippet = " ".join(str(raw_chunk).split())
        chunks.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "")),
                "source_file": str(metadata.get("source_file", "unknown")),
                "document_title": str(metadata.get("document_title", humanize_slug(Path(str(metadata.get("source_file", "unknown"))).stem))),
                "section_heading": str(metadata.get("section_heading", "Overview")),
                "content": str(raw_chunk),
                "snippet": snippet[:360] + ("..." if len(snippet) > 360 else ""),
                "score": float(score),
            }
        )

    return chunks


def retrieve_grounded_documents(
    patient_data: Mapping[str, Any] | None = None,
    question: str | None = None,
    prediction: Mapping[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    kb_dir: str | Path = KNOWLEDGE_BASE_DIR,
) -> list[dict[str, Any]]:
    """Compatibility wrapper used by existing app/tests to fetch semantic results."""
    persist_directory = None
    if Path(kb_dir) != KNOWLEDGE_BASE_DIR:
        persist_directory = VECTORSTORE_DIR / "tmp_test_chroma"
        build_vectorstore(kb_dir=kb_dir, persist_directory=persist_directory)

    query = build_retrieval_query(
        patient_data=patient_data,
        question=question,
        risk_prediction=str(prediction.get("risk_category")) if prediction else None,
        risk_factors=list(prediction.get("important_features", [])) if prediction else None,
    )
    chunks = retrieve_guideline_chunks(query=query, top_k=top_k, persist_directory=persist_directory)
    return [
        {
            "source": chunk["source_file"],
            "title": chunk["document_title"],
            "section_heading": chunk["section_heading"],
            "content": chunk["content"],
            "snippet": chunk["snippet"],
            "score": chunk["score"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]
