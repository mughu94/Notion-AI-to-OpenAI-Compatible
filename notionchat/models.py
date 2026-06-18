from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# Fallback when getAvailableModels is unavailable.
DEFAULT_MODEL_MAP: dict[str, str] = {
    "notion-ai": "ambrosia-tart-high",
    "gpt-4o": "ambrosia-tart-high",
    "gpt-4": "ambrosia-tart-high",
    "gpt-3.5-turbo": "almond-croissant-low",
    "gpt-5.2": "oatmeal-cookie",
    "gpt-5.4": "oval-kumquat-medium",
    "gpt-5.5": "opal-quince-medium",
    "opus-4.8": "ambrosia-tart-high",
    "opus-4.7": "apricot-sorbet-high",
    "opus-4.6": "avocado-froyo-medium",
    "sonnet-4.6": "almond-croissant-low",
    "haiku-4.5": "anthropic-haiku-4.5",
    "gemini-2.5-flash": "vertex-gemini-2.5-flash",
    "gemini-3-flash": "gingerbread",
    "minimax-m2.5": "fireworks-minimax-m2.5",
    "ambrosia-tart-high": "ambrosia-tart-high",
}

ANTHROPIC_ALIASES: dict[str, str] = {
    "claude-opus-4-7": "opus-4.7",
    "claude-opus-4-6": "opus-4.6",
    "claude-sonnet-4-6": "sonnet-4.6",
    "claude-haiku-4-5": "haiku-4.5",
}

_models_cache: tuple[float, list[dict[str, Any]], dict[str, str]] | None = None
MODELS_CACHE_TTL_SECONDS = 300.0


def friendly_alias(model_message: str) -> str:
    return model_message.strip().lower().replace(" ", "-")


def parse_available_models(response: dict[str, Any]) -> dict[str, str]:
    """Friendly alias -> Notion internal model id (with common short-name variants)."""
    out: dict[str, str] = {}
    for entry in response.get("models") or []:
        if not isinstance(entry, dict) or entry.get("isDisabled"):
            continue
        msg = entry.get("modelMessage")
        mid = entry.get("model")
        if not isinstance(msg, str) or not isinstance(mid, str):
            continue
        primary = friendly_alias(msg)
        if not primary:
            continue
        aliases = {primary, mid}
        if primary.startswith("claude-"):
            aliases.add(primary.removeprefix("claude-"))
        for short in (
            "opus-4.8",
            "opus-4.7",
            "opus-4.6",
            "sonnet-4.6",
            "haiku-4.5",
            "gemini-3-flash",
            "gemini-2.5-flash",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.2",
            "gpt-4o",
            "minimax-m2.5",
        ):
            if short in primary:
                aliases.add(short)
        for alias in aliases:
            out[alias] = mid
    return out


def _openai_model_entry(model_id: str, *, owned_by: str = "notion") -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": owned_by,
        "created": int(time.time()),
    }


def list_openai_models_from_notion(
    response: dict[str, Any],
    *,
    default_notion_id: str,
) -> list[dict[str, Any]]:
    """Build OpenAI /v1/models list — one entry per enabled Notion model only."""
    del default_notion_id  # kept for API compatibility

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_notion_ids: set[str] = set()

    for entry in response.get("models") or []:
        if not isinstance(entry, dict) or entry.get("isDisabled"):
            continue
        msg = entry.get("modelMessage")
        notion_id = entry.get("model")
        if not isinstance(msg, str) or not isinstance(notion_id, str):
            continue
        if notion_id in seen_notion_ids:
            continue

        model_id = friendly_alias(msg)
        if not model_id or model_id in seen_ids:
            continue

        seen_ids.add(model_id)
        seen_notion_ids.add(notion_id)
        models.append(_openai_model_entry(model_id))

    models.sort(key=lambda item: item["id"])
    return models


def list_openai_models(default: str) -> list[dict[str, Any]]:
    """Fallback when Notion model list cannot be fetched — no fabricated models."""
    del default
    return []


def cache_openai_models(
    models: list[dict[str, Any]],
    alias_map: dict[str, str] | None = None,
) -> None:
    global _models_cache
    _models_cache = (time.time(), models, alias_map or {})


def get_cached_openai_models() -> list[dict[str, Any]] | None:
    if _models_cache is None:
        return None
    cached_at, models, _ = _models_cache
    if time.time() - cached_at > MODELS_CACHE_TTL_SECONDS:
        return None
    return models


def get_cached_alias_map() -> dict[str, str] | None:
    if _models_cache is None:
        return None
    cached_at, _, alias_map = _models_cache
    if time.time() - cached_at > MODELS_CACHE_TTL_SECONDS:
        return None
    return alias_map or None


def normalize_request_model(model: str | None) -> str | None:
    """Strip router prefixes like `notion/opus-4.8` → `opus-4.8`."""
    if not model:
        return model
    cleaned = model.strip()
    while "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1].strip()
    return cleaned or model


def _lookup_model(name: str | None, mapping: dict[str, str]) -> str | None:
    if not name or not mapping:
        return None
    if name in mapping:
        return mapping[name]
    lower = name.lower().replace("_", "-")
    if lower in mapping:
        return mapping[lower]
    for alias, notion_id in mapping.items():
        if alias.lower() == lower:
            return notion_id
    if name in mapping.values():
        return name
    return None


def resolve_model(model: str | None, *, default: str, alias_map: dict[str, str] | None = None) -> str:
    dynamic = alias_map if alias_map is not None else (get_cached_alias_map() or {})
    model = normalize_request_model(model)
    default = normalize_request_model(default) or default

    if not model:
        return resolve_model(default, default=default, alias_map=alias_map)

    if model == "notion-ai":
        hit = _lookup_model("notion-ai", dynamic) or _lookup_model("notion-ai", DEFAULT_MODEL_MAP)
        return hit or resolve_model(default, default=default, alias_map=alias_map)

    hit = _lookup_model(model, dynamic)
    if hit:
        return hit

    hit = _lookup_model(model, DEFAULT_MODEL_MAP)
    if hit:
        return hit

    if model in ANTHROPIC_ALIASES:
        alias = ANTHROPIC_ALIASES[model]
        hit = _lookup_model(alias, dynamic) or _lookup_model(alias, DEFAULT_MODEL_MAP)
        if hit:
            return hit

    lower = (model or "").lower().replace("_", "-")
    if "opus" in lower:
        for key in ("opus-4.8", "opus-4.7", "opus-4.6"):
            if key in lower:
                hit = _lookup_model(key, dynamic) or _lookup_model(key, DEFAULT_MODEL_MAP)
                if hit:
                    return hit
    if "sonnet" in lower:
        hit = _lookup_model("sonnet-4.6", dynamic) or _lookup_model("sonnet-4.6", DEFAULT_MODEL_MAP)
        if hit:
            return hit
    if "haiku" in lower:
        hit = _lookup_model("haiku-4.5", dynamic) or _lookup_model("haiku-4.5", DEFAULT_MODEL_MAP)
        if hit:
            return hit

    log.warning("Unknown model %r — passing through to Notion", model)
    return model
