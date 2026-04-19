from __future__ import annotations

from typing import Any, Mapping

from src.agent.config import get_agent_config, validate_agent_config
from src.agent.prompts import CARDIO_SYSTEM_PROMPT, build_follow_up_prompt, build_guidance_prompt
from src.agent.retrieval import retrieve_grounded_documents
from src.features import coerce_and_validate_patient_payload, derive_risk_signals
from src.inference import load_artifact_bundle, predict_single


CARDIO_SCOPE_TERMS = {
    "blood",
    "pressure",
    "bp",
    "cholesterol",
    "diet",
    "exercise",
    "heart",
    "cardio",
    "smoking",
    "smoker",
    "diabetes",
    "angina",
    "chest pain",
    "risk",
    "stroke",
    "warning",
}

OUT_OF_SCOPE_TERMS = {
    "python",
    "javascript",
    "movie",
    "music",
    "football",
    "cricket",
    "politics",
    "election",
    "travel",
    "hotel",
    "stock",
    "crypto",
    "resume",
}

EMERGENCY_TERMS = {
    "chest pain",
    "shortness of breath",
    "fainting",
    "slurred speech",
    "face drooping",
    "one-sided weakness",
    "numbness",
}


def route_request(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the current task so the workflow stays domain-specific."""
    question = (state.get("question") or "").lower()

    if question and any(term in question for term in EMERGENCY_TERMS):
        route = "urgent_guidance"
    elif question and any(term in question for term in OUT_OF_SCOPE_TERMS):
        route = "domain_guardrail"
    elif question:
        route = "follow_up"
    else:
        route = "risk_assessment"

    state["route"] = route
    state["workflow_trace"].append("route_request")
    return state


def prepare_input(state: dict[str, Any]) -> dict[str, Any]:
    """Validate patient inputs and derive rule-based cardiovascular signals."""
    patient = coerce_and_validate_patient_payload(state["patient_data"])
    state["patient_profile"] = patient
    state["risk_signals"] = derive_risk_signals(patient)
    state["workflow_trace"].append("prepare_input")
    return state


def score_patient_risk(state: dict[str, Any]) -> dict[str, Any]:
    """Generate the ML risk score used by the rest of the workflow."""
    bundle = state.get("bundle") or load_artifact_bundle()
    state["bundle"] = bundle
    state["prediction"] = predict_single(state["patient_profile"], bundle=bundle)
    state["workflow_trace"].append("score_patient_risk")
    return state


def summarize_risk(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a patient-friendly interpretation of the model output."""
    prediction = state["prediction"]
    top_factors = prediction.get("important_features", [])[:3]
    factor_summary = ", ".join(item["feature"] for item in top_factors) if top_factors else "core clinical inputs"
    probability = prediction["probability"] * 100

    summary = (
        f"Estimated cardiovascular risk is {probability:.1f}%, which falls in the "
        f"{prediction['risk_category']} risk tier. The profile is most influenced by {factor_summary}."
    )
    state["summary"] = summary
    state["workflow_trace"].append("summarize_risk")
    return state


def retrieve_health_guidance(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve grounded recommendations from the markdown knowledge base."""
    state["kb_documents"] = retrieve_grounded_documents(
        patient_data=state["patient_profile"],
        question=state.get("question"),
        prediction=state["prediction"],
        top_k=4,
    )
    state["workflow_trace"].append("retrieve_health_guidance")
    return state


def _extract_action_items(document: Mapping[str, Any], limit: int = 2) -> list[str]:
    bullet_lines = []
    for line in str(document["content"]).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullet_lines.append(stripped[2:].strip())
    return bullet_lines[:limit]


def _rule_based_recommendations(state: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    patient = state["patient_profile"]

    if patient["systolic_bp"] >= 140:
        recommendations.append("Track home blood pressure regularly and discuss persistent readings above 140 mmHg with a clinician.")
    if patient["cholesterol"] >= 200:
        recommendations.append("Reduce saturated fat intake and prioritize fiber-rich meals to support cholesterol control.")
    if patient["bmi"] >= 25:
        recommendations.append("Aim for a steady, sustainable weight-management plan built around nutrition quality and regular activity.")
    if patient["smoker"] == "Yes":
        recommendations.append("Make smoking cessation a priority because it materially increases cardiovascular risk.")
    if patient["diabetes"] == "Yes":
        recommendations.append("Keep glucose management and medication adherence tightly coordinated with regular follow-up.")
    if patient["exercise_angina"] == "Yes":
        recommendations.append("Do not ignore exertional chest discomfort; seek medical review promptly if symptoms persist or worsen.")

    for document in state.get("kb_documents", []):
        recommendations.extend(_extract_action_items(document))

    deduplicated: list[str] = []
    seen = set()
    for recommendation in recommendations:
        lowered = recommendation.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduplicated.append(recommendation)
    return deduplicated[:6]


import streamlit as st

@st.cache_resource(show_spinner=False)
def _get_genai_client():
    from google import genai
    return genai.Client()

def _call_optional_llm(prompt: str) -> str | None:
    config = get_agent_config()
    if not config.llm_enabled:
        return None

    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=config.model_name,
            contents=f"{CARDIO_SYSTEM_PROMPT}\n\n{prompt}",
        )
        return (response.text or "").strip() or None
    except Exception:
        return None


def generate_recommendations(state: dict[str, Any]) -> dict[str, Any]:
    """Generate final grounded guidance, with optional LLM augmentation."""
    recommendations = _rule_based_recommendations(state)
    state["recommendations"] = recommendations

    llm_response = _call_optional_llm(build_guidance_prompt(state))
    if llm_response:
        state["guidance_note"] = llm_response
        state["generation_mode"] = "llm"
    else:
        source_titles = ", ".join(document["title"] for document in state.get("kb_documents", [])[:3])
        state["guidance_note"] = (
            f"{state['summary']} Recommended focus areas: "
            + " ".join(recommendations[:3])
            + (f" Guidance grounded in: {source_titles}." if source_titles else "")
        )
        state["generation_mode"] = "grounded_rules"

    state["workflow_trace"].append("generate_recommendations")
    return state


def validate_output(state: dict[str, Any]) -> dict[str, Any]:
    """Apply final domain guardrails before the result reaches the UI."""
    disclaimer = (
        "Educational support only. This tool is not a diagnosis and does not replace a licensed clinician."
    )
    urgent_note = ""

    if state["route"] == "urgent_guidance" or any(
        signal["id"] in {"exercise_angina", "chest_pain"} for signal in state.get("risk_signals", [])
    ):
        urgent_note = (
            " Seek urgent medical attention now for active chest pain, severe shortness of breath, fainting, "
            "or stroke-like symptoms."
        )

    if state["route"] == "domain_guardrail":
        state["follow_up_answer"] = (
            "I can answer follow-up questions only about cardiovascular risk, blood pressure, cholesterol, "
            "diet, exercise, warning signs, and follow-up monitoring."
        )

    state["disclaimer"] = disclaimer + urgent_note
    state["validated"] = True
    state["workflow_trace"].append("validate_output")
    return state


def answer_follow_up(state: dict[str, Any]) -> dict[str, Any]:
    """Answer a domain-specific follow-up question using retrieved guidance."""
    question = state.get("question")
    if not question:
        state["follow_up_answer"] = ""
        state["workflow_trace"].append("answer_follow_up")
        return state

    if state.get("route") == "domain_guardrail":
        state["workflow_trace"].append("answer_follow_up")
        return state

    llm_response = _call_optional_llm(build_follow_up_prompt(question, state))
    if llm_response:
        answer = llm_response
    else:
        snippets = " ".join(document["snippet"] for document in state.get("kb_documents", [])[:2])
        answer = (
            f"For this cardiovascular profile, the most relevant guidance is: {snippets} "
            f"Priority actions include {', '.join(state.get('recommendations', [])[:2]).lower()}."
        )

    state["follow_up_answer"] = answer
    state["workflow_trace"].append("answer_follow_up")
    return state


def run_agent_workflow(
    patient_data: Mapping[str, Any],
    question: str | None = None,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full step-based cardiovascular assistant workflow."""
    state: dict[str, Any] = {
        "patient_data": dict(patient_data),
        "question": (question or "").strip(),
        "bundle": dict(bundle) if bundle is not None else None,
        "workflow_trace": [],
        "config": validate_agent_config(),
    }

    for step in (
        route_request,
        prepare_input,
        score_patient_risk,
        summarize_risk,
        retrieve_health_guidance,
        generate_recommendations,
        validate_output,
        answer_follow_up,
    ):
        state = step(state)

    return {
        "route": state["route"],
        "summary": state["summary"],
        "prediction": state["prediction"],
        "risk_signals": state["risk_signals"],
        "recommendations": state["recommendations"],
        "guidance_note": state["guidance_note"],
        "follow_up_answer": state["follow_up_answer"],
        "kb_documents": [
            {
                "source": document["source"],
                "title": document["title"],
                "snippet": document["snippet"],
                "score": document["score"],
            }
            for document in state.get("kb_documents", [])
        ],
        "workflow_trace": state["workflow_trace"],
        "generation_mode": state["generation_mode"],
        "disclaimer": state["disclaimer"],
        "config": state["config"],
    }
