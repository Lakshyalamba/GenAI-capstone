from __future__ import annotations

from typing import Any, Mapping

from src.agent.state import RetrievedChunk


CARDIO_REPORT_SYSTEM_PROMPT = """
You are CardioRisk AI, a medically cautious cardiovascular health support assistant.
You are not allowed to invent diagnoses, fabricate guideline claims, or cite sources that were not provided.
Only ground your answer in:
1. the visible patient input
2. the model risk output
3. the retrieved guideline chunks

If data is incomplete, say so clearly.
If no guideline chunk was retrieved, say that explicitly and keep the response conservative.
Always include source attribution and always include the provided medical disclaimer.
Output only a structured markdown report with these exact sections:
## Risk Summary
## Key Factors
## Recommendations
## Follow-up Suggestions
## Sources
## Disclaimer
""".strip()


def _format_patient_context(patient_data: Mapping[str, Any]) -> str:
    if not patient_data:
        return "No normalized patient data is available."
    return "\n".join(f"- {key}: {value}" for key, value in patient_data.items())


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No guideline chunk was retrieved from the vector store."
    return "\n\n".join(
        (
            f"[{chunk['chunk_id']}] {chunk['document_title']} | "
            f"{chunk['source_file']} | {chunk['section_heading']}\n"
            f"{chunk['content']}"
        )
        for chunk in chunks
    )


def build_structured_report_prompt(state: Mapping[str, Any]) -> str:
    """Create the constrained report-generation prompt used for optional LLM polishing."""
    risk_factors = state.get("risk_factors", [])
    deterministic_recommendations = state.get("recommendations", [])
    follow_up_suggestions = state.get("follow_up_suggestions", [])

    factor_lines = "\n".join(
        f"- {factor.get('feature', 'Unknown factor')} ({factor.get('direction', 'context only')})"
        for factor in risk_factors
    ) or "- No model-derived risk factor explanation is available."

    recommendation_lines = "\n".join(f"- {item}" for item in deterministic_recommendations) or "- No deterministic recommendation was generated."
    follow_up_lines = "\n".join(f"- {item}" for item in follow_up_suggestions) or "- No follow-up suggestion was generated."
    source_lines = "\n".join(f"- {item}" for item in state.get("retrieved_sources", [])) or "- No guideline chunk was retrieved."

    return f"""
Patient input:
{_format_patient_context(state.get("normalized_patient_data", {}))}

Missing fields:
{", ".join(state.get("missing_fields", [])) or "None"}

Risk label:
{state.get("risk_prediction") or "Unavailable"}

Risk probability:
{state.get("risk_probability") if state.get("risk_probability") is not None else "Unavailable"}

Draft summary:
{state.get("draft_summary", "")}

Model-extracted key factors:
{factor_lines}

Deterministic recommendations:
{recommendation_lines}

Follow-up suggestions:
{follow_up_lines}

Retrieved guideline chunks:
{_format_chunks(state.get("retrieved_chunks", []))}

Source labels:
{source_lines}

Medical disclaimer:
{state.get("disclaimer", "")}

Additional user focus:
{state.get("question", "") or "None"}
""".strip()
