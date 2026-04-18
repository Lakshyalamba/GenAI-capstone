from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass

from src.utils import KNOWLEDGE_BASE_DIR


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model_name: str
    api_key_present: bool
    openai_package_available: bool
    llm_enabled: bool
    fallback_mode: str
    knowledge_base_dir: str


def get_agent_config() -> AgentConfig:
    """Build the effective agent configuration from environment state."""
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    openai_package_available = importlib.util.find_spec("openai") is not None
    llm_enabled = api_key_present and openai_package_available

    return AgentConfig(
        provider="openai",
        model_name=os.getenv("CARDIO_AGENT_MODEL", "gpt-4.1-mini"),
        api_key_present=api_key_present,
        openai_package_available=openai_package_available,
        llm_enabled=llm_enabled,
        fallback_mode="grounded-rules",
        knowledge_base_dir=str(KNOWLEDGE_BASE_DIR),
    )


def validate_agent_config() -> dict[str, object]:
    """Return a UI-friendly status payload for agent startup checks."""
    config = get_agent_config()
    issues: list[str] = []

    if not config.api_key_present:
        issues.append("OPENAI_API_KEY is not set; using grounded rule-based recommendations.")
    if config.api_key_present and not config.openai_package_available:
        issues.append("The `openai` package is unavailable; using grounded rule-based recommendations.")

    status = "llm_enabled" if config.llm_enabled else "fallback_only"

    payload = asdict(config)
    payload.update(
        {
            "status": status,
            "issues": issues,
            "fallback_available": True,
        }
    )
    return payload
