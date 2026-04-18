from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def ensure_project_directories() -> None:
    """Create the runtime directories expected by the project."""
    for path in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, KNOWLEDGE_BASE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _to_builtin(value: Any) -> Any:
    """Convert numpy and pandas scalar objects into JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_json(path: Path, default: Any | None = None) -> Any:
    """Load JSON content, returning a default value when the file is absent."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    """Persist JSON content with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_builtin(data), file, indent=2, sort_keys=True)


def humanize_slug(value: str) -> str:
    """Convert a snake_case or kebab-case token into title case."""
    return value.replace("_", " ").replace("-", " ").strip().title()


def get_env_status() -> dict[str, Any]:
    """Expose runtime environment details for the UI and diagnostics."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "agent_model": os.getenv("CARDIO_AGENT_MODEL", "gpt-4.1-mini"),
        "app_env": os.getenv("APP_ENV", "local"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
    }
