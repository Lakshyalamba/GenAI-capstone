from __future__ import annotations

from pathlib import Path

from src.agent.retrieval import (
    build_retrieval_query,
    build_vectorstore,
    load_knowledge_base,
    retrieve_guideline_chunks,
)


class DeterministicKeywordEmbeddings:
    """Simple local embeddings used to keep the retrieval tests offline and deterministic."""

    vocabulary = [
        "blood",
        "pressure",
        "monitoring",
        "cuff",
        "home",
        "sodium",
        "warning",
        "chest",
        "pain",
        "diet",
        "cholesterol",
        "exercise",
        "follow",
        "smoking",
        "diabetes",
    ]

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(token)) for token in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def sample_patient() -> dict[str, object]:
    return {
        "age": 61,
        "systolic_bp": 164,
        "cholesterol": 252,
        "max_heart_rate": 118,
        "bmi": 32.2,
        "sex": "Female",
        "chest_pain": "typical",
        "smoker": "Yes",
        "diabetes": "Yes",
        "exercise_angina": "Yes",
    }


def test_knowledge_base_loads_markdown_files() -> None:
    documents = load_knowledge_base()
    sources = {document["source"] for document in documents}
    assert "diet_guidance.md" in sources
    assert "warning_signs.md" in sources


def test_retrieval_query_includes_risk_label_and_patient_context() -> None:
    query = build_retrieval_query(
        patient_data=sample_patient(),
        question="What should this patient do about blood pressure?",
        risk_prediction="High",
        risk_factors=[{"feature": "Systolic Blood Pressure (164)", "direction": "raises risk"}],
    )
    assert "Predicted cardiovascular risk tier: High." in query
    assert "Clinical focus: What should this patient do about blood pressure?" in query
    assert "Systolic Blood Pressure" in query


def test_chroma_retrieval_finds_blood_pressure_guidance(tmp_path: Path) -> None:
    embeddings = DeterministicKeywordEmbeddings()
    persist_dir = tmp_path / "chroma_db"
    build_vectorstore(persist_directory=persist_dir, embeddings=embeddings)

    documents = retrieve_guideline_chunks(
        query="blood pressure monitoring and home cuff guidance",
        top_k=3,
        persist_directory=persist_dir,
        embeddings=embeddings,
    )
    sources = {document["source_file"] for document in documents}
    assert "bp_management.md" in sources


def test_chroma_retrieval_finds_warning_signs_guidance(tmp_path: Path) -> None:
    embeddings = DeterministicKeywordEmbeddings()
    persist_dir = tmp_path / "warning_db"
    build_vectorstore(persist_directory=persist_dir, embeddings=embeddings)

    documents = retrieve_guideline_chunks(
        query="severe chest pain emergency warning signs",
        top_k=2,
        persist_directory=persist_dir,
        embeddings=embeddings,
    )
    sources = {document["source_file"] for document in documents}
    assert "warning_signs.md" in sources
