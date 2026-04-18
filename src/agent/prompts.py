from __future__ import annotations

from typing import Any


CARDIO_SYSTEM_PROMPT = """
You are CardioRisk AI, a cardiovascular risk support assistant.
Use only the supplied model summary and retrieved knowledge-base excerpts.
Do not invent medical facts, do not provide emergency diagnosis, and keep the answer concise,
actionable, and limited to cardiovascular risk, prevention, monitoring, diet, exercise,
blood pressure, cholesterol, smoking cessation, and warning signs.
Always acknowledge that the output is educational support and not a medical diagnosis.
""".strip()


def build_guidance_prompt(state: dict[str, Any]) -> str:
    """Create the grounded recommendation prompt for optional LLM augmentation."""
    retrieved_context = "\n\n".join(
        f"[{doc['source']}]\n{doc['snippet']}" for doc in state.get("kb_documents", [])
    )
    recommendation_seed = "\n".join(f"- {item}" for item in state.get("recommendations", []))
    return f"""
Prediction summary:
{state.get("summary", "")}

Risk category: {state.get("prediction", {}).get("risk_category", "Unknown")}
Probability: {state.get("prediction", {}).get("probability", 0.0):.2%}

Retrieved knowledge:
{retrieved_context}

Draft recommendations:
{recommendation_seed}

Write a short cardiovascular guidance note grounded in the retrieved knowledge.
""".strip()


def build_follow_up_prompt(question: str, state: dict[str, Any]) -> str:
    """Create the grounded follow-up prompt for optional LLM augmentation."""
    retrieved_context = "\n\n".join(
        f"[{doc['source']}]\n{doc['snippet']}" for doc in state.get("kb_documents", [])
    )
    return f"""
Question: {question}

Patient risk summary:
{state.get("summary", "")}

Retrieved knowledge:
{retrieved_context}

Answer only within the cardiovascular prevention domain.
""".strip()
