from __future__ import annotations

from src.agent.retrieval import load_knowledge_base, retrieve_grounded_documents


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


def test_retrieval_finds_blood_pressure_guidance() -> None:
    documents = retrieve_grounded_documents(
        patient_data=sample_patient(),
        question="What should this patient do about blood pressure?",
        top_k=3,
    )
    sources = {document["source"] for document in documents}
    assert "bp_management.md" in sources


def test_retrieval_finds_warning_signs_for_emergency_question() -> None:
    documents = retrieve_grounded_documents(
        question="When is chest pain an emergency?",
        top_k=2,
    )
    sources = {document["source"] for document in documents}
    assert "warning_signs.md" in sources
