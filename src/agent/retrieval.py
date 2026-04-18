from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.features import derive_risk_signals
from src.utils import KNOWLEDGE_BASE_DIR, humanize_slug


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase keywords for simple retrieval."""
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def _parse_document(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), humanize_slug(path.stem))
    keyword_line = next((line for line in lines if line.lower().startswith("keywords:")), "Keywords:")
    keywords = {
        keyword.strip().lower()
        for keyword in keyword_line.split(":", maxsplit=1)[1].split(",")
        if keyword.strip()
    }
    tokens = tokenize(content) | keywords
    return {
        "source": path.name,
        "title": title,
        "content": content,
        "keywords": keywords,
        "tokens": tokens,
    }


def load_knowledge_base(kb_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, Any]]:
    """Load all knowledge-base markdown documents."""
    base_dir = Path(kb_dir)
    documents = sorted(base_dir.glob("*.md"))
    return [_parse_document(path) for path in documents]


def build_query_terms(
    patient_data: dict[str, Any] | None = None,
    question: str | None = None,
    prediction: dict[str, Any] | None = None,
) -> set[str]:
    """Combine patient signals and user intent into retrieval terms."""
    terms = set()
    if question:
        terms |= tokenize(question)

    if patient_data:
        for signal in derive_risk_signals(patient_data):
            terms |= tokenize(signal["label"])
            terms |= {topic.lower() for topic in signal["topics"]}

    if prediction:
        risk_category = prediction.get("risk_category")
        if risk_category == "High":
            terms |= {"preventive", "warning", "monitoring"}
        elif risk_category == "Moderate":
            terms |= {"lifestyle", "diet", "exercise"}

    return terms


def _extract_snippet(content: str, query_terms: set[str], max_lines: int = 3) -> str:
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.lower().startswith("keywords:")
    ]
    scored_lines = []
    for line in lines:
        overlap = len(tokenize(line) & query_terms)
        if overlap:
            scored_lines.append((overlap, line))

    if scored_lines:
        selected = [line for _, line in sorted(scored_lines, key=lambda item: item[0], reverse=True)[:max_lines]]
    else:
        selected = lines[:max_lines]

    return " ".join(selected)


def retrieve_grounded_documents(
    patient_data: dict[str, Any] | None = None,
    question: str | None = None,
    prediction: dict[str, Any] | None = None,
    top_k: int = 3,
    kb_dir: str | Path = KNOWLEDGE_BASE_DIR,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant knowledge-base documents for the current case."""
    question_terms = tokenize(question or "")
    profile_terms = build_query_terms(patient_data=patient_data, question=None, prediction=prediction)
    query_terms = question_terms | profile_terms
    documents = load_knowledge_base(kb_dir=kb_dir)
    ranked_documents = []

    for document in documents:
        overlap_score = len(document["tokens"] & profile_terms)
        keyword_bonus = len(document["keywords"] & profile_terms) * 2
        question_overlap = len(document["tokens"] & question_terms) * 3
        question_keyword_bonus = len(document["keywords"] & question_terms) * 4
        source_bonus = 0
        if "warning" in query_terms and document["source"] == "warning_signs.md":
            source_bonus = 3
        if {"blood", "pressure"} & question_terms and document["source"] == "bp_management.md":
            source_bonus += 5
        score = overlap_score + keyword_bonus + question_overlap + question_keyword_bonus + source_bonus
        if score == 0 and not query_terms:
            score = 1

        ranked_documents.append(
            {
                **document,
                "score": score,
                "snippet": _extract_snippet(document["content"], query_terms),
            }
        )

    ranked_documents.sort(key=lambda item: (item["score"], item["source"]), reverse=True)
    non_zero = [document for document in ranked_documents if document["score"] > 0]
    return (non_zero or ranked_documents)[:top_k]
