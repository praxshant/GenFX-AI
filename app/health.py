"""
Runtime health probes for the sidebar.

Every probe is cheap and non-destructive - no image is generated, no GPU
seconds are spent. Nothing here raises.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app import config
from app.blend_builder import describe_runtime

logger = logging.getLogger(__name__)

TIMEOUT = 6


def _ok(detail: str) -> dict[str, Any]:
    return {"ok": True, "detail": detail, "error": None}


def _bad(error: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "detail": detail, "error": error}


def _key_probe(url: str, key: str) -> str | None:
    """None when the key is accepted, else a short reason. Spends no tokens."""
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    except Exception as exc:
        return f"unreachable: {str(exc)[:60]}"
    return None if resp.status_code == 200 else f"HTTP {resp.status_code}"


def _probe_llm(name: str) -> tuple[str | None, str | None]:
    """(label if usable, reason if not) for one provider in the cascade."""
    if name == "openrouter":
        if not config.OPENROUTER_API_KEY:
            return None, None
        why = _key_probe("https://openrouter.ai/api/v1/key", config.OPENROUTER_API_KEY)
        return (None, f"OpenRouter {why}") if why else (f"OpenRouter · {config.OPENROUTER_MODEL}", None)

    if name == "openai":
        if not config.OPENAI_API_KEY:
            return None, None
        why = _key_probe("https://api.openai.com/v1/models", config.OPENAI_API_KEY)
        return (None, f"OpenAI {why}") if why else (f"OpenAI · {config.OPENAI_MODEL}", None)

    if name == "huggingface":
        if not config.HUGGINGFACE_API_KEY:
            return None, None
        why = _key_probe("https://huggingface.co/api/whoami-v2", config.HUGGINGFACE_API_KEY)
        return (None, f"HuggingFace {why}") if why else (f"HuggingFace · {config.HF_LLM_MODEL}", None)

    if name == "ollama":
        if not config.OLLAMA_ENABLED:
            return None, None
        from app.llm_parser import _ollama_has_model

        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
            installed = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception:
            return None, "Ollama not running"
        if not _ollama_has_model(config.OLLAMA_MODEL, installed):
            return None, f"Ollama has no '{config.OLLAMA_MODEL}'"
        return f"Ollama · {config.OLLAMA_MODEL}", None

    return None, None


def check_llm() -> dict[str, Any]:
    """
    The provider the parser will actually land on, walking the same cascade it
    does - so a rejected key upstream shows as a fallback, not a dead end.
    """
    skipped: list[str] = []
    for name in config.LLM_PROVIDER_ORDER:
        label, why = _probe_llm(name)
        if label:
            return _ok(label + (f" (after: {'; '.join(skipped)})" if skipped else ""))
        if why:
            skipped.append(why)

    detail = "local heuristic parser will be used"
    if skipped:
        return _bad("; ".join(skipped), detail)
    return _bad("No LLM provider configured", detail)


def check_image() -> dict[str, Any]:
    """Pollinations needs no key, so this is usually green out of the box."""
    if config.POLLINATIONS_ENABLED and "pollinations" in config.IMAGE_PROVIDER_ORDER:
        try:
            resp = requests.head(
                f"{config.POLLINATIONS_ENDPOINT}/ping?width=64&height=64",
                headers={"User-Agent": "GenFX/2.0"},
                timeout=TIMEOUT,
                allow_redirects=True,
            )
            if resp.status_code < 500:
                return _ok(f"Pollinations · {config.POLLINATIONS_MODEL} (keyless)")
            return _bad(f"Pollinations HTTP {resp.status_code}")
        except Exception as exc:
            if config.HUGGINGFACE_API_KEY:
                return _ok("HuggingFace Inference (Pollinations unreachable)")
            return _bad(f"Pollinations unreachable: {str(exc)[:80]}")

    if config.HUGGINGFACE_API_KEY:
        return _ok(f"HuggingFace · {config.HF_IMAGE_MODELS[0]}")
    return _bad("No image provider configured")


def check_mesh() -> dict[str, Any]:
    """Is at least one image-to-3D Space actually up right now?"""
    for space_id in config.MESH_SPACES:
        try:
            resp = requests.get(
                f"https://huggingface.co/api/spaces/{space_id}",
                timeout=TIMEOUT,
                headers={"User-Agent": "GenFX/2.0"},
            )
            if resp.status_code != 200:
                continue
            stage = (resp.json().get("runtime") or {}).get("stage")
            if stage == "RUNNING":
                suffix = "" if config.HUGGINGFACE_API_KEY else " (anonymous quota)"
                return _ok(f"{space_id}{suffix}")
        except Exception:
            continue
    return _bad(
        "No image-to-3D Space is running",
        "local inflation mesh will be used instead",
    )


def check_blend() -> dict[str, Any]:
    """A .blend runtime is the one thing the product cannot do without."""
    runtime = describe_runtime()
    if runtime.get("blender_binary"):
        return _ok(f"Blender · {runtime['blender_binary']}")
    if runtime.get("bpy_module"):
        return _ok("bpy module (no Blender install needed)")
    if runtime.get("worker"):
        return _ok(f"remote worker · {runtime['worker']}")
    return _bad("No Blender runtime", "install `bpy` or set BLENDER_PATH")


def check_assets() -> dict[str, Any]:
    if not config.FALLBACK_IMAGE_PATH.exists():
        return _bad(
            f"missing: {config.FALLBACK_IMAGE_PATH.name}", "run python create_fallback_assets.py"
        )
    return _ok("fallback assets present")


def check_runtime_health() -> dict[str, dict[str, Any]]:
    """Every probe, in one dict. Safe to call at app startup."""
    probes = {
        "llm": check_llm,
        "image": check_image,
        "mesh": check_mesh,
        "blend": check_blend,
        "assets": check_assets,
    }
    out: dict[str, dict[str, Any]] = {}
    for name, probe in probes.items():
        try:
            out[name] = probe()
        except Exception as exc:  # a probe must never take down the page
            out[name] = _bad(f"probe crashed: {type(exc).__name__} - {str(exc)[:80]}")
    return out
