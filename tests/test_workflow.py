from __future__ import annotations

from src.agent import workflow as workflow_module


def sample_patient() -> dict[str, object]:
    return {
        "age": 58,
        "systolic_bp": 148,
        "cholesterol": 245,
        "max_heart_rate": 132,
        "bmi": 31.4,
        "sex": "Male",
        "chest_pain": "asymptomatic",
        "smoker": "Yes",
        "diabetes": "No",
        "exercise_angina": "No",
    }


def fake_chunk() -> dict[str, object]:
    return {
        "chunk_id": "bp_management-001",
        "source_file": "bp_management.md",
        "document_title": "Blood Pressure Management",
        "section_heading": "Key actions",
        "content": "- Use a validated home blood pressure cuff.\n- Reduce sodium intake.",
        "snippet": "Use a validated home blood pressure cuff. Reduce sodium intake.",
        "score": 0.12,
    }


def test_workflow_runs_end_to_end_with_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(workflow_module, "retrieve_guideline_chunks", lambda query, top_k=4: [fake_chunk()])
    monkeypatch.setattr(workflow_module, "_call_optional_llm", lambda prompt: None)
    workflow_module.get_compiled_workflow.cache_clear()

    result = workflow_module.run_agent_workflow(
        patient_data=sample_patient(),
        question="Prioritize immediate lifestyle interventions.",
    )

    assert result["partial_output"] is False
    assert result["risk_probability"] is not None
    assert result["retrieved_sources"]
    assert "## Risk Summary" in result["final_report"]["rendered_markdown"]
    assert any("score_risk" in step for step in result["workflow_trace"])


def test_workflow_returns_partial_report_when_required_fields_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(workflow_module, "retrieve_guideline_chunks", lambda query, top_k=4: [])
    monkeypatch.setattr(workflow_module, "_call_optional_llm", lambda prompt: None)
    workflow_module.get_compiled_workflow.cache_clear()

    patient = sample_patient()
    patient.pop("cholesterol")

    result = workflow_module.run_agent_workflow(patient_data=patient)

    assert result["partial_output"] is True
    assert "cholesterol" in result["missing_fields"]
    assert result["prediction"]["risk_category"] == "Unavailable"
    assert result["fallback_status"]
    assert "## Disclaimer" in result["final_report"]["rendered_markdown"]
