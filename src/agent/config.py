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
    sdk_package_available: bool
    llm_enabled: bool
    fallback_mode: str
    knowledge_base_dir: str


def _package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _read_secret(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except Exception:
        return None

    return None


def get_agent_config() -> AgentConfig:
    """Build the effective agent configuration from environment state."""
    api_key_present = bool(_read_secret("GEMINI_API_KEY"))
    sdk_package_available = _package_available("google.genai")
    llm_enabled = api_key_present and sdk_package_available

    return AgentConfig(
        provider="gemini",
        model_name=_read_secret("CARDIO_AGENT_MODEL") or "gemini-2.5-flash",
        api_key_present=api_key_present,
        sdk_package_available=sdk_package_available,
        llm_enabled=llm_enabled,
        fallback_mode="grounded-rules",
        knowledge_base_dir=str(KNOWLEDGE_BASE_DIR),
    )


def validate_agent_config() -> dict[str, object]:
    """Return a UI-friendly status payload for agent startup checks."""
    config = get_agent_config()
    issues: list[str] = []

    if not config.api_key_present:
        issues.append("GEMINI_API_KEY is not set; using grounded rule-based recommendations.")
    if config.api_key_present and not config.sdk_package_available:
        issues.append("The `google-genai` package is unavailable; using grounded rule-based recommendations.")

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
