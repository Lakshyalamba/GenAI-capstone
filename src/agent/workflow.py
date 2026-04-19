from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from src.agent.config import get_agent_config, validate_agent_config
from src.agent.prompts import CARDIO_REPORT_SYSTEM_PROMPT, build_structured_report_prompt
from src.agent.retrieval import (
    build_retrieval_query,
    format_retrieved_sources,
    retrieve_guideline_chunks,
)
from src.agent.state import AgentState, StructuredHealthReport
from src.features import derive_partial_risk_signals, inspect_patient_payload
from src.inference import load_artifact_bundle, predict_single

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError as exc:  # pragma: no cover - exercised through graceful fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


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


def _ensure_langgraph_dependencies() -> None:
    if _LANGGRAPH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LangGraph is unavailable. Install the project requirements before running the agent workflow."
        ) from _LANGGRAPH_IMPORT_ERROR


def _question_contains(question: str, vocabulary: set[str]) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in vocabulary)


def _build_disclaimer(state: Mapping[str, Any]) -> str:
    disclaimer = (
        "Educational support only. This assistant does not diagnose disease and does not replace a licensed clinician."
    )
    if _question_contains(state.get("question", ""), EMERGENCY_TERMS) or any(
        signal.get("id") in {"exercise_angina", "chest_pain"} for signal in state.get("risk_signals", [])
    ):
        disclaimer += (
            " Seek urgent medical attention now for severe chest pain, shortness of breath, fainting, or stroke-like symptoms."
        )
    return disclaimer


def _format_probability(probability: float | None) -> str:
    return f"{probability:.1%}" if probability is not None else "Unavailable"


def _extract_chunk_actions(chunk: Mapping[str, Any], limit: int = 2) -> list[str]:
    actions: list[str] = []
    for line in str(chunk.get("content", "")).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            actions.append(stripped[2:].strip())
    return actions[:limit]


def _rule_based_recommendations(state: Mapping[str, Any]) -> list[str]:
    recommendations: list[str] = []
    patient = state.get("normalized_patient_data", {})
    question = state.get("question", "")

    if _question_contains(question, OUT_OF_SCOPE_TERMS):
        recommendations.append(
            "This assistant is limited to cardiovascular risk support, prevention, monitoring, lifestyle counseling, and warning signs."
        )

    systolic_bp = patient.get("systolic_bp")
    if systolic_bp is not None and systolic_bp >= 140:
        recommendations.append(
            "Track home blood pressure regularly and discuss persistent readings above 140 mmHg with a clinician."
        )

    cholesterol = patient.get("cholesterol")
    if cholesterol is not None and cholesterol >= 200:
        recommendations.append(
            "Reduce saturated fat intake, improve fiber intake, and review cholesterol trends with a clinician."
        )

    bmi = patient.get("bmi")
    if bmi is not None and bmi >= 25:
        recommendations.append(
            "Use a gradual weight-management plan built around nutrition quality and consistent physical activity."
        )

    if patient.get("smoker") == "Yes":
        recommendations.append(
            "Smoking cessation should be a near-term priority because it materially increases cardiovascular risk."
        )

    if patient.get("diabetes") == "Yes":
        recommendations.append(
            "Coordinate glucose control, medication adherence, and regular follow-up because diabetes compounds cardiovascular risk."
        )

    if patient.get("exercise_angina") == "Yes":
        recommendations.append(
            "Do not ignore exertional chest discomfort; seek prompt medical review if symptoms persist, worsen, or are active now."
        )

    for chunk in state.get("retrieved_chunks", []):
        recommendations.extend(_extract_chunk_actions(chunk))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        normalized = recommendation.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(recommendation)

    if not deduplicated:
        deduplicated.append(
            "Use conservative cardiovascular prevention steps and obtain clinician follow-up if symptoms or risk markers worsen."
        )

    return deduplicated[:6]


def _build_follow_up_suggestions(state: Mapping[str, Any]) -> list[str]:
    suggestions: list[str] = []
    if state.get("missing_fields"):
        suggestions.append(
            "Complete the missing patient fields before relying on the ML risk score."
        )
    if state.get("risk_probability") is None:
        suggestions.append(
            "Re-run the assessment after completing the patient profile so the model can produce a calibrated risk estimate."
        )
    else:
        suggestions.append("Repeat this assessment when new vitals or lab values are available.")

    if state.get("risk_prediction") in {"Moderate", "High"}:
        suggestions.append(
            "Discuss the overall cardiovascular risk profile and preventive options with a licensed clinician."
        )

    if any(signal.get("id") == "blood_pressure" for signal in state.get("risk_signals", [])):
        suggestions.append("Keep a short home blood-pressure log for follow-up review.")

    if any(signal.get("id") == "cholesterol" for signal in state.get("risk_signals", [])):
        suggestions.append("Review a repeat lipid panel or trend data during clinical follow-up.")

    return suggestions[:4]


def _build_draft_summary(state: Mapping[str, Any]) -> str:
    probability = state.get("risk_probability")
    risk_prediction = state.get("risk_prediction")
    missing_fields = state.get("missing_fields", [])
    question = state.get("question", "").strip()
    factors = [factor.get("feature", "Unknown factor") for factor in state.get("risk_factors", [])[:3]]

    if probability is not None and risk_prediction:
        summary = (
            f"The ML model estimates a {_format_probability(probability)} cardiovascular risk probability, "
            f"which falls in the {risk_prediction} risk tier."
        )
    else:
        summary = (
            "A reliable ML risk score could not be produced because the patient profile is incomplete or a downstream tool failed."
        )

    if factors:
        summary += " The most visible drivers in this case are " + ", ".join(factors) + "."

    if missing_fields:
        summary += " Missing fields: " + ", ".join(missing_fields) + "."

    if not state.get("retrieved_chunks"):
        summary += " No guideline chunk was retrieved, so the response remains conservative and source-limited."

    if question:
        summary += f" Clinical focus requested: {question}."

    if _question_contains(question, OUT_OF_SCOPE_TERMS):
        summary += " The question included non-cardiovascular content, so the answer is constrained to this project's health-support scope."

    return summary


def _render_report_markdown(report: StructuredHealthReport) -> str:
    factor_lines = "\n".join(f"- {item}" for item in report["key_factors"]) or "- No model-derived factor is available."
    recommendation_lines = "\n".join(f"- {item}" for item in report["recommendations"])
    follow_up_lines = "\n".join(f"- {item}" for item in report["follow_up_suggestions"])
    source_lines = "\n".join(f"- {item}" for item in report["sources"])
    return "\n\n".join(
        [
            "## Risk Summary\n" + report["risk_summary"],
            "## Key Factors\n" + factor_lines,
            "## Recommendations\n" + recommendation_lines,
            "## Follow-up Suggestions\n" + follow_up_lines,
            "## Sources\n" + source_lines,
            "## Disclaimer\n" + report["disclaimer"],
        ]
    )


def _call_optional_llm(prompt: str) -> str | None:
    config = get_agent_config()
    if not config.llm_enabled:
        return None

    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model=config.model_name,
            contents=f"{CARDIO_REPORT_SYSTEM_PROMPT}\n\n{prompt}",
        )
        return (response.text or "").strip() or None
    except Exception:
        return None


def validate_input(state: AgentState) -> dict[str, Any]:
    """Inspect the incoming patient payload without crashing on incomplete input."""
    inspected = inspect_patient_payload(state.get("patient_input", {}))
    fallback_status: list[str] = []

    if inspected["missing_fields"]:
        fallback_status.append(
            "Partial report generated because one or more required patient fields are missing."
        )
    if inspected["errors"]:
        fallback_status.append(
            "Partial report generated because one or more patient fields were invalid for model scoring."
        )
    if _question_contains(state.get("question", ""), OUT_OF_SCOPE_TERMS):
        fallback_status.append(
            "The user question included out-of-scope content; the assistant response is limited to cardiovascular support."
        )

    return {
        "missing_fields": inspected["missing_fields"],
        "errors": inspected["errors"],
        "partial_output": bool(inspected["missing_fields"] or inspected["errors"]),
        "fallback_status": fallback_status,
        "workflow_trace": [
            f"validate_input: {len(inspected['missing_fields'])} missing field(s), {len(inspected['errors'])} validation issue(s)"
        ],
    }


def normalize_input(state: AgentState) -> dict[str, Any]:
    """Normalize any valid patient fields and derive rule-based clinical signals."""
    inspected = inspect_patient_payload(state.get("patient_input", {}))
    normalized = dict(inspected["normalized_data"])
    return {
        "normalized_patient_data": normalized,
        "risk_signals": derive_partial_risk_signals(normalized),
        "workflow_trace": [f"normalize_input: normalized {len(normalized)} patient field(s)"],
    }


def _route_after_normalization(state: AgentState) -> str:
    return "handle_fallback" if state.get("missing_fields") or state.get("errors") else "score_risk"


def score_risk(state: AgentState) -> dict[str, Any]:
    """Run the existing ML pipeline when the full patient payload is valid."""
    try:
        bundle = state.get("bundle") or load_artifact_bundle()
        prediction = predict_single(state["normalized_patient_data"], bundle=bundle)
        return {
            "bundle": bundle,
            "risk_prediction": prediction["risk_category"],
            "predicted_class": prediction["predicted_class"],
            "risk_probability": prediction["probability"],
            "risk_factors": prediction["important_features"],
            "workflow_trace": ["score_risk: ML risk score generated"],
        }
    except Exception as exc:
        return {
            "errors": [f"Risk scoring failed: {exc}"],
            "partial_output": True,
            "fallback_status": [
                "ML scoring failed; the agent continued with a deterministic partial report."
            ],
            "workflow_trace": ["score_risk: ML scoring failed; routing to fallback"],
        }


def _route_after_scoring(state: AgentState) -> str:
    return "extract_risk_factors" if state.get("risk_probability") is not None else "handle_fallback"


def extract_risk_factors(state: AgentState) -> dict[str, Any]:
    """Trim the explainability payload to the highest-signal factors for downstream use."""
    factors = list(state.get("risk_factors", []))[:5]
    return {
        "risk_factors": factors,
        "workflow_trace": [f"extract_risk_factors: prepared {len(factors)} factor(s)"],
    }


def handle_fallback(state: AgentState) -> dict[str, Any]:
    """Create a safe partial state when the workflow cannot produce a full ML-backed report."""
    rule_based_factors = list(state.get("risk_factors", []))
    if not rule_based_factors:
        rule_based_factors = [
            {
                "feature": signal["label"],
                "direction": "raises risk" if signal.get("severity") in {"high", "moderate"} else "context only",
                "contribution": 0.0,
            }
            for signal in state.get("risk_signals", [])[:5]
        ]

    fallback_updates: list[str] = []
    if state.get("risk_probability") is None and not state.get("fallback_status"):
        fallback_updates.append(
            "The workflow could not produce a full ML-backed report, so it returned a deterministic partial report."
        )

    return {
        "risk_prediction": state.get("risk_prediction") or "Unavailable",
        "risk_probability": state.get("risk_probability"),
        "risk_factors": rule_based_factors,
        "partial_output": True,
        "fallback_status": fallback_updates,
        "workflow_trace": ["handle_fallback: prepared graceful partial-output state"],
    }


def retrieve_guidelines(state: AgentState) -> dict[str, Any]:
    """Retrieve semantically relevant guidance chunks from the local Chroma store."""
    query = build_retrieval_query(
        patient_data=state.get("normalized_patient_data"),
        question=state.get("question"),
        risk_prediction=state.get("risk_prediction"),
        risk_factors=state.get("risk_factors"),
    )

    try:
        chunks = retrieve_guideline_chunks(query=query, top_k=4)
        sources = format_retrieved_sources(chunks)
        updates: dict[str, Any] = {
            "retrieval_query": query,
            "retrieved_chunks": chunks,
            "retrieved_sources": sources or ["No guideline chunk was retrieved from the vector store."],
            "workflow_trace": [f"retrieve_guidelines: retrieved {len(chunks)} chunk(s) from Chroma"],
        }
        if not chunks:
            updates["fallback_status"] = [
                "No guideline chunk was retrieved; the response is limited to patient inputs and deterministic safety guidance."
            ]
        return updates
    except Exception as exc:
        return {
            "retrieval_query": query,
            "retrieved_chunks": [],
            "retrieved_sources": ["No guideline chunk was retrieved from the vector store."],
            "errors": [f"Guideline retrieval failed: {exc}"],
            "partial_output": True,
            "fallback_status": [
                "Guideline retrieval failed; the response uses only visible patient data and deterministic safeguards."
            ],
            "workflow_trace": ["retrieve_guidelines: retrieval failed; continuing with safe fallback"],
        }


def generate_summary(state: AgentState) -> dict[str, Any]:
    """Write a deterministic grounded summary for the current workflow state."""
    return {
        "disclaimer": _build_disclaimer(state),
        "draft_summary": _build_draft_summary(state),
        "workflow_trace": ["generate_summary: built deterministic draft summary"],
    }


def generate_recommendations(state: AgentState) -> dict[str, Any]:
    """Generate deterministic recommendations and optionally polish the report with an LLM."""
    recommendations = _rule_based_recommendations(state)
    follow_up_suggestions = _build_follow_up_suggestions(state)

    updates: dict[str, Any] = {
        "recommendations": recommendations,
        "follow_up_suggestions": follow_up_suggestions,
        "generation_mode": "deterministic_report",
        "workflow_trace": ["generate_recommendations: built deterministic recommendations"],
    }

    prompt = build_structured_report_prompt(
        {
            **state,
            "recommendations": recommendations,
            "follow_up_suggestions": follow_up_suggestions,
        }
    )
    llm_report = _call_optional_llm(prompt)
    if llm_report:
        updates["llm_report"] = llm_report
        updates["generation_mode"] = "llm_structured_report"
        updates["workflow_trace"] = [
            "generate_recommendations: LLM produced a structured report grounded in retrieved chunks"
        ]
    elif state.get("config", {}).get("llm_enabled"):
        updates["fallback_status"] = [
            "The LLM report-generation step failed, so the assistant used the deterministic report renderer."
        ]

    return updates


def validate_output(state: AgentState) -> dict[str, Any]:
    """Assemble the final structured report and enforce the output contract."""
    disclaimer = state.get("disclaimer") or _build_disclaimer(state)
    key_factors = [factor.get("feature", "Unknown factor") for factor in state.get("risk_factors", [])]
    sources = state.get("retrieved_sources") or ["No guideline chunk was retrieved from the vector store."]
    report: StructuredHealthReport = {
        "risk_summary": state.get("draft_summary", ""),
        "key_factors": key_factors or ["No model-derived key factor is available."],
        "recommendations": state.get("recommendations", []),
        "follow_up_suggestions": state.get("follow_up_suggestions", []),
        "sources": sources,
        "disclaimer": disclaimer,
        "rendered_markdown": "",
    }

    deterministic_markdown = _render_report_markdown(report)
    rendered_markdown = deterministic_markdown
    llm_report = state.get("llm_report")
    required_sections = [
        "## Risk Summary",
        "## Key Factors",
        "## Recommendations",
        "## Follow-up Suggestions",
        "## Sources",
        "## Disclaimer",
    ]
    if llm_report and all(section in llm_report for section in required_sections):
        rendered_markdown = llm_report
    elif llm_report:
        report["rendered_markdown"] = deterministic_markdown
        return {
            "final_report": report,
            "follow_up_answer": report["risk_summary"]
            + " "
            + " ".join(report["recommendations"][:2])
            + f" Sources: {', '.join(report['sources'][:2])}.",
            "generation_mode": "deterministic_report",
            "fallback_status": [
                "The LLM output did not follow the required report format, so the deterministic report renderer was used."
            ],
            "workflow_trace": ["validate_output: assembled deterministic structured report"],
            "disclaimer": disclaimer,
        }

    report["rendered_markdown"] = rendered_markdown
    follow_up_answer = report["risk_summary"]
    if report["recommendations"]:
        follow_up_answer += " Priority recommendations: " + "; ".join(report["recommendations"][:2]) + "."
    if report["sources"]:
        follow_up_answer += " Sources: " + "; ".join(report["sources"][:2]) + "."

    return {
        "final_report": report,
        "follow_up_answer": follow_up_answer,
        "workflow_trace": ["validate_output: assembled final structured report"],
        "disclaimer": disclaimer,
    }


@lru_cache(maxsize=1)
def get_compiled_workflow():
    """Build and cache the LangGraph StateGraph used by the app."""
    _ensure_langgraph_dependencies()
    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("score_risk", score_risk)
    graph.add_node("extract_risk_factors", extract_risk_factors)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("generate_recommendations", generate_recommendations)
    graph.add_node("validate_output", validate_output)
    graph.add_node("handle_fallback", handle_fallback)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "normalize_input")
    graph.add_conditional_edges(
        "normalize_input",
        _route_after_normalization,
        {
            "score_risk": "score_risk",
            "handle_fallback": "handle_fallback",
        },
    )
    graph.add_conditional_edges(
        "score_risk",
        _route_after_scoring,
        {
            "extract_risk_factors": "extract_risk_factors",
            "handle_fallback": "handle_fallback",
        },
    )
    graph.add_edge("extract_risk_factors", "retrieve_guidelines")
    graph.add_edge("handle_fallback", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "generate_summary")
    graph.add_edge("generate_summary", "generate_recommendations")
    graph.add_edge("generate_recommendations", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile(name="agentic_health_support_assistant")


def _build_graceful_failure_result(
    patient_data: Mapping[str, Any],
    question: str,
    bundle: Mapping[str, Any] | None,
    error: Exception,
) -> dict[str, Any]:
    state: AgentState = {
        "patient_input": dict(patient_data),
        "question": question,
        "bundle": dict(bundle) if bundle is not None else None,
        "config": validate_agent_config(),
        "missing_fields": [],
        "normalized_patient_data": {},
        "risk_prediction": "Unavailable",
        "predicted_class": None,
        "risk_probability": None,
        "risk_factors": [],
        "risk_signals": [],
        "retrieved_chunks": [],
        "retrieved_sources": ["No guideline chunk was retrieved from the vector store."],
        "retrieval_query": "",
        "draft_summary": "The workflow encountered an internal failure before a full report could be produced.",
        "recommendations": [
            "Review the application logs, verify the dependency setup, and re-run the assessment.",
            "Use clinician judgment instead of relying on this incomplete output.",
        ],
        "follow_up_suggestions": [
            "Complete the dependency setup and rerun the LangGraph workflow.",
        ],
        "final_report": {
            "risk_summary": "The workflow encountered an internal failure before a full report could be produced.",
            "key_factors": ["No reliable factor summary is available."],
            "recommendations": [
                "Review the application logs, verify the dependency setup, and re-run the assessment.",
                "Use clinician judgment instead of relying on this incomplete output.",
            ],
            "follow_up_suggestions": ["Complete the dependency setup and rerun the LangGraph workflow."],
            "sources": ["No guideline chunk was retrieved from the vector store."],
            "disclaimer": "Educational support only. This assistant does not diagnose disease and does not replace a licensed clinician.",
            "rendered_markdown": "",
        },
        "follow_up_answer": "",
        "disclaimer": "Educational support only. This assistant does not diagnose disease and does not replace a licensed clinician.",
        "llm_report": None,
        "generation_mode": "graceful_failure",
        "partial_output": True,
        "errors": [str(error)],
        "fallback_status": [
            "The LangGraph workflow failed unexpectedly, so a minimal safe fallback report was returned."
        ],
        "workflow_trace": [f"graceful_failure: {error}"],
    }
    state["final_report"]["rendered_markdown"] = _render_report_markdown(state["final_report"])
    state["follow_up_answer"] = state["final_report"]["risk_summary"]
    return _format_workflow_result(state)


def _format_workflow_result(state: Mapping[str, Any]) -> dict[str, Any]:
    final_report = dict(state.get("final_report", {}))
    prediction = {
        "predicted_class": state.get("predicted_class"),
        "probability": state.get("risk_probability"),
        "risk_category": state.get("risk_prediction") or "Unavailable",
        "important_features": state.get("risk_factors", []),
    }
    return {
        "route": "partial_report" if state.get("partial_output") else "full_report",
        "summary": state.get("draft_summary", ""),
        "prediction": prediction,
        "risk_prediction": state.get("risk_prediction"),
        "risk_probability": state.get("risk_probability"),
        "risk_signals": state.get("risk_signals", []),
        "missing_fields": state.get("missing_fields", []),
        "retrieval_query": state.get("retrieval_query", ""),
        "recommendations": state.get("recommendations", []),
        "guidance_note": final_report.get("risk_summary", state.get("draft_summary", "")),
        "follow_up_answer": state.get("follow_up_answer", ""),
        "kb_documents": [
            {
                "source": chunk["source_file"],
                "title": chunk["document_title"],
                "section_heading": chunk["section_heading"],
                "snippet": chunk["snippet"],
                "score": chunk["score"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in state.get("retrieved_chunks", [])
        ],
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "retrieved_sources": state.get("retrieved_sources", []),
        "workflow_trace": state.get("workflow_trace", []),
        "generation_mode": state.get("generation_mode", "deterministic_report"),
        "fallback_status": state.get("fallback_status", []),
        "partial_output": bool(state.get("partial_output")),
        "errors": state.get("errors", []),
        "disclaimer": state.get("disclaimer", ""),
        "config": state.get("config", {}),
        "final_report": final_report,
    }


def run_agent_workflow(
    patient_data: Mapping[str, Any],
    question: str | None = None,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the LangGraph-based cardiovascular support workflow."""
    state: AgentState = {
        "patient_input": dict(patient_data),
        "question": (question or "").strip(),
        "bundle": dict(bundle) if bundle is not None else None,
        "config": validate_agent_config(),
        "missing_fields": [],
        "normalized_patient_data": {},
        "risk_prediction": None,
        "predicted_class": None,
        "risk_probability": None,
        "risk_factors": [],
        "risk_signals": [],
        "retrieved_chunks": [],
        "retrieved_sources": [],
        "retrieval_query": "",
        "draft_summary": "",
        "recommendations": [],
        "follow_up_suggestions": [],
        "final_report": {
            "risk_summary": "",
            "key_factors": [],
            "recommendations": [],
            "follow_up_suggestions": [],
            "sources": [],
            "disclaimer": "",
            "rendered_markdown": "",
        },
        "follow_up_answer": "",
        "disclaimer": "",
        "llm_report": None,
        "generation_mode": "deterministic_report",
        "partial_output": False,
        "errors": [],
        "fallback_status": [],
        "workflow_trace": [],
    }

    try:
        compiled_workflow = get_compiled_workflow()
        result_state = compiled_workflow.invoke(state)
        return _format_workflow_result(result_state)
    except Exception as exc:
        return _build_graceful_failure_result(patient_data, (question or "").strip(), bundle, exc)
