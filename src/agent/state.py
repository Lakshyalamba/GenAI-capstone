from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RetrievedChunk(TypedDict):
    """A single retrieved knowledge chunk with display metadata."""

    chunk_id: str
    source_file: str
    document_title: str
    section_heading: str
    content: str
    snippet: str
    score: float


class StructuredHealthReport(TypedDict):
    """Structured report payload rendered in the UI."""

    risk_summary: str
    key_factors: list[str]
    recommendations: list[str]
    follow_up_suggestions: list[str]
    sources: list[str]
    disclaimer: str
    rendered_markdown: str


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the health support assistant."""

    patient_input: dict[str, Any]
    question: str
    bundle: dict[str, Any] | None
    config: dict[str, Any]
    missing_fields: list[str]
    normalized_patient_data: dict[str, Any]
    risk_prediction: str | None
    predicted_class: int | None
    risk_probability: float | None
    risk_factors: list[dict[str, Any]]
    risk_signals: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    retrieved_sources: list[str]
    retrieval_query: str
    draft_summary: str
    recommendations: list[str]
    follow_up_suggestions: list[str]
    final_report: StructuredHealthReport
    follow_up_answer: str
    disclaimer: str
    llm_report: str | None
    generation_mode: str
    partial_output: bool
    errors: Annotated[list[str], operator.add]
    fallback_status: Annotated[list[str], operator.add]
    workflow_trace: Annotated[list[str], operator.add]
