"""LLM model selection service with runtime and .env persistence."""

from __future__ import annotations

from pathlib import Path

from src.config import AVAILABLE_LLM_MODELS, PROJECT_ROOT, get_llm_model, set_llm_model


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


def get_models() -> dict[str, object]:
    """Return the current model and selectable model options."""
    current = get_llm_model()
    available = _parse_available_models(AVAILABLE_LLM_MODELS)
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
