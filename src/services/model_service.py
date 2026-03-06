"""LLM model selection service with runtime and .env persistence."""

from __future__ import annotations

from pathlib import Path

from ollama import Client

from src.config import AVAILABLE_LLM_MODELS, OLLAMA_BASE_URL, PROJECT_ROOT, get_llm_model, set_llm_model


def _parse_available_models(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    # Keep order and uniqueness.
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _fetch_ollama_models() -> list[str]:
    """Fetch currently available model names from the running Ollama instance."""
    try:
        response = Client(host=OLLAMA_BASE_URL).list()
    except Exception:  # noqa: BLE001
        return []

    if isinstance(response, dict):
        raw_models = response.get("models", [])
    else:
        raw_models = getattr(response, "models", [])

    names: list[str] = []
    for item in raw_models:
        name = ""
        if isinstance(item, dict):
            name = str(item.get("model") or item.get("name") or "").strip()
        else:
            name = str(getattr(item, "model", "") or getattr(item, "name", "")).strip()
        if name:
            names.append(name)
    return _parse_available_models(",".join(names))


def get_models() -> dict[str, object]:
    """Return the current model and selectable model options."""
    current = get_llm_model().strip()
    from_ollama = _fetch_ollama_models()
    from_env = _parse_available_models(AVAILABLE_LLM_MODELS)

    available = list(from_ollama)
    for model_name in from_env:
        if model_name not in available:
            available.append(model_name)

    if current not in available:
        available.append(current)
    return {"current": current, "available": available}


def select_model(model: str) -> dict[str, object]:
    """Validate, persist and apply the selected model."""
    model_name = model.strip()
    if not model_name:
        raise ValueError("Model name cannot be empty")

    models_data = get_models()
    available = models_data["available"]
    if model_name not in available:
        raise ValueError(f"Model '{model_name}' is not in available models")

    _write_llm_model_to_env(model_name, PROJECT_ROOT / ".env")
    set_llm_model(model_name)
    return get_models()


def _write_llm_model_to_env(model: str, env_path: Path) -> None:
    line_value = f"LLM_MODEL={model}"
    if not env_path.exists():
        env_path.write_text(f"{line_value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("LLM_MODEL="):
            updated.append(line_value)
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(line_value)

    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
