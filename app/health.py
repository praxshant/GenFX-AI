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


def check_llm() -> dict[str, Any]:
    """Which parser provider is reachable, without spending a completion."""
    if config.OPENROUTER_API_KEY:
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                return _ok(f"OpenRouter · {config.OPENROUTER_MODEL}")
            return _bad(f"OpenRouter HTTP {resp.status_code}")
        except Exception as exc:
            return _bad(f"OpenRouter unreachable: {str(exc)[:80]}")

    if config.OPENAI_API_KEY:
        return _ok(f"OpenAI · {config.OPENAI_MODEL}")

    if config.HUGGINGFACE_API_KEY:
        return _ok(f"HuggingFace · {config.HF_LLM_MODEL}")

    if config.OLLAMA_ENABLED:
        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])][:3]
                return _ok(f"Ollama · {', '.join(models) or config.OLLAMA_MODEL}")
        except Exception:
            pass

    return _bad("No LLM provider configured", "local heuristic parser will be used")


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
    missing = [
        p.name
        for p in (config.FALLBACK_JSON_PATH, config.FALLBACK_IMAGE_PATH, config.FALLBACK_RENDER_PATH)
        if not p.exists()
    ]
    if missing:
        return _bad(f"missing: {', '.join(missing)}", "run python create_fallback_assets.py")
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
