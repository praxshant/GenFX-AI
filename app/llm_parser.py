"""
Stage 1 - Prompt parser.

Turns a free-text description into a validated Scene JSON that the rest of the
pipeline actually consumes: `image_prompt` drives image generation, and
`lighting` / `camera` / `materials` drive the Blender scene that gets saved.

Providers are tried in order (OpenRouter -> OpenAI -> HuggingFace -> Ollama).
A provider that is unconfigured is skipped; a provider that errors falls
through to the next one. If every provider fails, a deterministic local
heuristic parser still produces a usable scene from the raw prompt, so this
stage never blocks the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app import config

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "scene_id",
    "subject",
    "environment",
    "lighting",
    "camera",
    "materials",
    "effects",
    "render_settings",
    "image_prompt",
    "asset_refs",
]

SCENE_SCHEMA_EXAMPLE = """{
  "scene_id": "sc_001",
  "subject": {
    "name": "vintage red sports car",
    "category": "product",
    "description": "a low-slung two-door coupe with chrome trim"
  },
  "environment": {
    "type": "studio",
    "time_of_day": "neutral"
  },
  "lighting": {
    "preset": "three_point_studio",
    "intensity": 1.0,
    "key_color": "#ffffff"
  },
  "camera": {
    "shot_type": "three_quarter",
    "angle": "eye_level",
    "focal_length": 50
  },
  "materials": {
    "base_color": "#c0392b",
    "roughness": 0.35,
    "metallic": 0.1
  },
  "effects": [
    { "type": "soft_shadow", "density": "low" }
  ],
  "render_settings": {
    "resolution": "1920x1080",
    "samples": 64
  },
  "image_prompt": "a vintage red sports car, three quarter view, studio lighting, seamless white background, product photography, sharp focus, high detail",
  "asset_refs": []
}"""

SYSTEM_PROMPT = f"""You are a 3D asset pipeline data formatter.
Convert the user's description into a single JSON object describing one clear
subject that can be generated as a standalone 3D asset.

RULES
1. Return ONLY valid JSON. No prose, no markdown, no code fences.
2. Match the schema exactly - same top-level keys, no extras, none missing.
3. "image_prompt" must be a rich text-to-image prompt describing ONE isolated
   subject on a plain, uncluttered background. Always append terms that help
   3D reconstruction: centered, full object visible, plain background,
   even lighting, no cropping.
4. "materials.base_color" and "lighting.key_color" must be hex colours.
5. Keep "asset_refs" an empty list.
6. Use snake_case for enum-like string values.

SCHEMA
{SCENE_SCHEMA_EXAMPLE}"""

STRICT_JSON_SUFFIX = (
    "\n\nRespond with ONLY the JSON object. "
    "Start your response with { and end with }."
)

# Terms appended to every image prompt: single centered object on a plain
# background is what image-to-3D reconstruction needs to work well.
RECON_SUFFIX = (
    "centered composition, full subject visible, plain seamless background, "
    "even diffuse lighting, no cropping, high detail, photorealistic"
)


class ValidationError(Exception):
    """Raised when a parsed dict does not match the required scene schema."""


class ProviderUnavailable(Exception):
    """Raised when a provider is not configured or not reachable."""


@dataclass
class ParserResult:
    scene_json: dict[str, Any]
    status: str  # "ok" | "degraded" | "fallback"
    error_message: str | None = None
    provider_used: str | None = None
    attempts: list[str] = field(default_factory=list)


# ── Validation & normalisation ────────────────────────────────────────────────

def validate_schema(data: dict) -> bool:
    """Validate required keys and nested types. Raises ValidationError."""
    if not isinstance(data, dict):
        raise ValidationError("Scene must be a JSON object.")

    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValidationError(f"Missing required key: '{key}'")

    for key in ("subject", "environment", "lighting", "camera", "materials", "render_settings"):
        if not isinstance(data.get(key), dict):
            raise ValidationError(f"'{key}' must be a dict.")

    for key in ("effects", "asset_refs"):
        if not isinstance(data.get(key), list):
            raise ValidationError(f"'{key}' must be a list.")

    if not isinstance(data.get("image_prompt"), str) or not data["image_prompt"].strip():
        raise ValidationError("'image_prompt' must be a non-empty string.")

    intensity = data["lighting"].get("intensity")
    if intensity is not None and not isinstance(intensity, (int, float)):
        raise ValidationError("'lighting.intensity' must be numeric.")

    return True


_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _clean_hex(value: Any, default: str) -> str:
    if isinstance(value, str) and _HEX_RE.match(value.strip()):
        v = value.strip()
        if len(v) == 4:  # #abc -> #aabbcc
            v = "#" + "".join(c * 2 for c in v[1:])
        return v.lower()
    return default


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def normalise_scene(data: dict, user_prompt: str = "") -> dict:
    """
    Fill in defaults and coerce types so downstream stages can trust the scene.
    Accepts partial or legacy scene dicts.
    """
    out = dict(data or {})

    out.setdefault("scene_id", "sc_001")

    subject = out.get("subject")
    if not isinstance(subject, dict):
        subject = {"name": str(subject) if subject else (user_prompt.strip() or "subject")}
    subject.setdefault("name", user_prompt.strip() or "subject")
    subject.setdefault("category", "product")
    subject.setdefault("description", subject["name"])
    out["subject"] = subject

    env = out.get("environment") if isinstance(out.get("environment"), dict) else {}
    env.setdefault("type", "studio")
    env.setdefault("time_of_day", "neutral")
    out["environment"] = env

    light = out.get("lighting") if isinstance(out.get("lighting"), dict) else {}
    light.setdefault("preset", "three_point_studio")
    light["intensity"] = _clamp(light.get("intensity", 1.0), 0.05, 5.0, 1.0)
    light["key_color"] = _clean_hex(light.get("key_color"), "#ffffff")
    out["lighting"] = light

    cam = out.get("camera") if isinstance(out.get("camera"), dict) else {}
    cam.setdefault("shot_type", "three_quarter")
    cam.setdefault("angle", "eye_level")
    cam["focal_length"] = _clamp(cam.get("focal_length", 50), 12, 300, 50)
    out["camera"] = cam

    mat = out.get("materials") if isinstance(out.get("materials"), dict) else {}
    mat["base_color"] = _clean_hex(mat.get("base_color"), "#b8b8b8")
    mat["roughness"] = _clamp(mat.get("roughness", 0.45), 0.0, 1.0, 0.45)
    mat["metallic"] = _clamp(mat.get("metallic", 0.0), 0.0, 1.0, 0.0)
    out["materials"] = mat

    effects = out.get("effects")
    out["effects"] = [e for e in effects if isinstance(e, dict)] if isinstance(effects, list) else []

    rs = out.get("render_settings") if isinstance(out.get("render_settings"), dict) else {}
    rs.setdefault("resolution", "1920x1080")
    try:
        rs["samples"] = int(rs.get("samples", 64))
    except (TypeError, ValueError):
        rs["samples"] = 64
    rs["samples"] = max(4, min(512, rs["samples"]))
    out["render_settings"] = rs

    prompt = out.get("image_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = build_heuristic_image_prompt(user_prompt or subject["name"])
    prompt = prompt.strip()
    if RECON_SUFFIX.split(",")[0] not in prompt:
        prompt = f"{prompt}, {RECON_SUFFIX}"
    out["image_prompt"] = prompt

    out["asset_refs"] = []
    return out


def build_heuristic_image_prompt(user_prompt: str) -> str:
    """Deterministic prompt builder used when no LLM is reachable."""
    base = (user_prompt or "an object").strip().rstrip(".")
    return f"{base}, single subject, studio product shot, {RECON_SUFFIX}"


def load_fallback_json() -> dict:
    """Load the pre-baked fallback scene, or rebuild it from the schema."""
    try:
        with open(config.FALLBACK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.loads(SCENE_SCHEMA_EXAMPLE)


# ── JSON extraction ───────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: str) -> dict:
    """
    Pull a JSON object out of raw model output.

    Handles markdown fences, leading prose, and trailing commentary. Tries the
    largest balanced {...} span first, then progressively smaller candidates.
    """
    if not raw or not raw.strip():
        raise json.JSONDecodeError("Empty model response", raw or "", 0)

    text = raw.strip()

    candidates: list[str] = []

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # Balanced-brace scan: collect every top-level {...} span in the text.
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])

    # Last resort: naive first-brace to last-brace slice.
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    seen: set[str] = set()
    errors: list[str] = []
    for cand in sorted(candidates, key=len, reverse=True):
        if cand in seen:
            continue
        seen.add(cand)
        for attempt in (cand, _repair_json(cand)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

    raise json.JSONDecodeError(
        f"No parseable JSON object found ({errors[0] if errors else 'no candidates'})",
        text,
        0,
    )


def _repair_json(text: str) -> str:
    """Fix the two things small models get wrong most: trailing commas and
    single-quoted keys."""
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    repaired = re.sub(r"'([^'\"\n]*)'(\s*:)", r'"\1"\2', repaired)
    return repaired


# ── Providers ─────────────────────────────────────────────────────────────────

def _chat_via_http(endpoint: str, headers: dict, model: str, user_prompt: str,
                   temperature: float = 0.1, max_tokens: int = 900) -> str:
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + STRICT_JSON_SUFFIX},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        endpoint, headers=headers, json=payload, timeout=config.API_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat response shape: {str(data)[:200]}") from exc


def _parse_with_openrouter(user_prompt: str) -> str:
    if not config.OPENROUTER_API_KEY:
        raise ProviderUnavailable("OPENROUTER_API_KEY is not set.")
    return _chat_via_http(
        config.OPENROUTER_ENDPOINT,
        {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/praxshant/GenFX-Lite",
            "X-Title": "GenFX",
        },
        config.OPENROUTER_MODEL,
        user_prompt,
    )


def _parse_with_openai(user_prompt: str) -> str:
    if not config.OPENAI_API_KEY:
        raise ProviderUnavailable("OPENAI_API_KEY is not set.")
    return _chat_via_http(
        config.OPENAI_ENDPOINT,
        {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        config.OPENAI_MODEL,
        user_prompt,
    )


def _parse_with_huggingface(user_prompt: str) -> str:
    """HuggingFace Inference Providers - OpenAI-compatible router endpoint."""
    if not config.HUGGINGFACE_API_KEY:
        raise ProviderUnavailable("HUGGINGFACE_API_KEY is not set.")
    return _chat_via_http(
        "https://router.huggingface.co/v1/chat/completions",
        {
            "Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        },
        config.HF_LLM_MODEL,
        user_prompt,
    )


def _parse_with_ollama(user_prompt: str) -> str:
    """
    Local Ollama daemon. Nothing is installed or downloaded by GenFX - if the
    daemon isn't running on OLLAMA_HOST this raises ProviderUnavailable and the
    cascade continues.
    """
    import requests

    if not config.OLLAMA_ENABLED:
        raise ProviderUnavailable("Ollama disabled via OLLAMA_ENABLED=0.")

    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        tags.raise_for_status()
        installed = {m.get("name", "") for m in tags.json().get("models", [])}
    except Exception as exc:
        raise ProviderUnavailable(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}") from exc

    model = config.OLLAMA_MODEL
    if installed and not any(name.split(":")[0] == model.split(":")[0] for name in installed):
        raise ProviderUnavailable(
            f"Ollama model '{model}' not pulled. Run: ollama pull {model}"
        )

    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + STRICT_JSON_SUFFIX},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=config.API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# Names, not function objects: resolving at call time keeps providers
# swappable (and mockable) instead of frozen at import.
PROVIDERS: dict[str, str] = {
    "openrouter": "_parse_with_openrouter",
    "openai": "_parse_with_openai",
    "huggingface": "_parse_with_huggingface",
    "ollama": "_parse_with_ollama",
}


def get_provider(name: str) -> Callable[[str], str] | None:
    """Resolve a provider by name at call time."""
    attr = PROVIDERS.get(name)
    return globals().get(attr) if attr else None


def available_providers() -> list[str]:
    """Providers that look configured, in cascade order."""
    ready = []
    for name in config.LLM_PROVIDER_ORDER:
        if name == "openrouter" and config.OPENROUTER_API_KEY:
            ready.append(name)
        elif name == "openai" and config.OPENAI_API_KEY:
            ready.append(name)
        elif name == "huggingface" and config.HUGGINGFACE_API_KEY:
            ready.append(name)
        elif name == "ollama" and config.OLLAMA_ENABLED:
            ready.append(name)
    return ready


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_prompt(user_prompt: str) -> ParserResult:
    """
    Parse a natural-language description into a normalised Scene JSON.

    Walks the provider cascade; each provider gets LLM_RETRY_COUNT + 1 tries.
    Falls back to a deterministic local scene rather than failing.
    """
    attempts: list[str] = []
    last_error = ""

    for provider_name in config.LLM_PROVIDER_ORDER:
        provider = get_provider(provider_name)
        if provider is None:
            continue

        for attempt in range(config.LLM_RETRY_COUNT + 1):
            try:
                raw = provider(user_prompt)
                parsed = _extract_json(raw)
                scene = normalise_scene(parsed, user_prompt)
                validate_schema(scene)
                logger.info("Scene parsed via %s (scene_id=%s)", provider_name, scene["scene_id"])
                attempts.append(f"{provider_name}: ok")
                return ParserResult(
                    scene_json=scene,
                    status="ok",
                    provider_used=provider_name,
                    attempts=attempts,
                )
            except ProviderUnavailable as exc:
                last_error = f"{provider_name}: {exc}"
                attempts.append(last_error)
                logger.debug("Provider %s unavailable: %s", provider_name, exc)
                break  # no point retrying an unconfigured provider
            except json.JSONDecodeError as exc:
                last_error = f"{provider_name}: invalid JSON ({exc.msg})"
                attempts.append(last_error)
                logger.warning("%s (attempt %d)", last_error, attempt + 1)
            except ValidationError as exc:
                last_error = f"{provider_name}: schema mismatch ({exc})"
                attempts.append(last_error)
                logger.warning("%s (attempt %d)", last_error, attempt + 1)
            except Exception as exc:
                last_error = f"{provider_name}: {type(exc).__name__} - {str(exc)[:160]}"
                attempts.append(last_error)
                logger.warning("%s (attempt %d)", last_error, attempt + 1)

    # Every provider failed - build a scene locally so the pipeline continues.
    logger.warning("All LLM providers failed; using local heuristic scene.")
    scene = normalise_scene(
        {
            "scene_id": "sc_local",
            "subject": {"name": user_prompt.strip() or "subject", "category": "product"},
            "image_prompt": build_heuristic_image_prompt(user_prompt),
        },
        user_prompt,
    )
    return ParserResult(
        scene_json=scene,
        status="fallback",
        error_message=last_error or "No LLM provider configured.",
        provider_used="local_heuristic",
        attempts=attempts,
    )
