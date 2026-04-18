from __future__ import annotations

from src.agent.config import validate_agent_config


def test_agent_config_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    status = validate_agent_config()
    assert status["api_key_present"] is False
    assert status["status"] == "fallback_only"
    assert status["fallback_available"] is True


def test_agent_config_with_api_key_is_consistent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    status = validate_agent_config()
    assert status["api_key_present"] is True
    assert status["llm_enabled"] is status["openai_package_available"]
    expected_status = "llm_enabled" if status["openai_package_available"] else "fallback_only"
    assert status["status"] == expected_status
