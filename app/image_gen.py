"""
Stage 2 - Scene JSON -> reference image.

Providers in cascade order:
  1. Pollinations   - free, keyless, FLUX-backed. This is what keeps the
                      public demo working with zero configuration.
  2. HuggingFace    - Inference Providers, used when a token is present.
  3. Fallback asset - pre-baked placeholder so the pipeline never stalls.

The image is post-processed for 3D reconstruction: square canvas, subject
centred, plain background. Image-to-3D models are far more reliable when fed
a single centred object.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from app import config

logger = logging.getLogger(__name__)

USER_AGENT = "GenFX/2.0 (+https://github.com/praxshant/GenFX-Lite)"


@dataclass
class ImageResult:
    image_path: str
    status: str  # "ok" | "fallback"
    provider_used: str | None = None
    error_message: str | None = None
    attempts: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0


def build_image_prompt(scene_json: dict[str, Any], user_prompt: str | None = None) -> str:
    """
    Return the text prompt for image generation.

    Preference order: the LLM-authored `image_prompt` (richest), then the raw
    user prompt, then a prompt assembled from the structured scene fields.
    """
    if isinstance(scene_json, dict):
        authored = scene_json.get("image_prompt")
        if isinstance(authored, str) and authored.strip():
            return authored.strip()

    if user_prompt and user_prompt.strip():
        from app.llm_parser import build_heuristic_image_prompt

        return build_heuristic_image_prompt(user_prompt)

    scene_json = scene_json or {}
    subject = (scene_json.get("subject") or {}).get("name", "")
    env = (scene_json.get("environment") or {}).get("type", "")
    tod = (scene_json.get("environment") or {}).get("time_of_day", "")
    light = (scene_json.get("lighting") or {}).get("preset", "")
    shot = (scene_json.get("camera") or {}).get("shot_type", "")
    fx = ", ".join(
        e.get("type", "")
        for e in (scene_json.get("effects") or [])
        if isinstance(e, dict) and e.get("type")
    )
    parts = [
        subject or f"cinematic {env or 'studio'} scene",
        tod.replace("_", " ") if tod else "",
        light.replace("_", " ") if light else "",
        f"{shot.replace('_', ' ')} shot" if shot else "",
        fx,
        "single subject, plain background, photorealistic, high detail",
    ]
    return ", ".join(p for p in parts if p)


# ── Providers ─────────────────────────────────────────────────────────────────

def _generate_pollinations(prompt: str, width: int, height: int, seed: int | None) -> bytes:
    """Keyless free generation. Returns raw image bytes."""
    import requests

    if not config.POLLINATIONS_ENABLED:
        raise RuntimeError("Pollinations disabled.")

    params = {
        "width": width,
        "height": height,
        "nologo": "true",
        "model": config.POLLINATIONS_MODEL,
        "enhance": "false",
    }
    if seed is not None:
        params["seed"] = seed

    url = (
        f"{config.POLLINATIONS_ENDPOINT}/{urllib.parse.quote(prompt[:1800])}"
        f"?{urllib.parse.urlencode(params)}"
    )
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
        timeout=config.IMAGE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    if not resp.content or len(resp.content) < 1024:
        raise RuntimeError(f"Pollinations returned {len(resp.content)} bytes")
    return resp.content


def _generate_huggingface(prompt: str, width: int, height: int, seed: int | None) -> bytes:
    """HuggingFace Inference Providers via huggingface_hub."""
    from huggingface_hub import InferenceClient

    if not config.HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY missing")

    client = InferenceClient(token=config.HUGGINGFACE_API_KEY, timeout=config.IMAGE_TIMEOUT_SECONDS)
    last: Exception | None = None
    for model in config.HF_IMAGE_MODELS:
        try:
            pil = client.text_to_image(prompt, model=model, width=width, height=height)
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:  # try the next model
            last = exc
            logger.warning("HF model %s failed: %s", model, str(exc)[:160])
    raise RuntimeError(f"All HF image models failed: {last}")


# Names, not function objects - resolved at call time so providers stay
# swappable and mockable.
PROVIDERS: dict[str, str] = {
    "pollinations": "_generate_pollinations",
    "huggingface": "_generate_huggingface",
}


def get_provider(name: str):
    attr = PROVIDERS.get(name)
    return globals().get(attr) if attr else None


# ── Post-processing for 3D reconstruction ─────────────────────────────────────

def prepare_for_reconstruction(img: Image.Image, size: int = 1024) -> Image.Image:
    """
    Square-pad the image on a background sampled from its own corners, so the
    subject stays centred and uncropped. Image-to-3D models expect this.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img.resize((size, size), Image.LANCZOS)

    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    bg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))

    side = max(w, h)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    # Soften the seam between padding and image edge.
    canvas = canvas.filter(ImageFilter.SMOOTH)
    return canvas.resize((size, size), Image.LANCZOS)


def generate_image(
    scene_json: dict[str, Any],
    output_path: str | Path,
    user_prompt: str | None = None,
    seed: int | None = None,
) -> ImageResult:
    """
    Generate the reference image, writing a PNG to `output_path`.

    Never raises: on total failure it returns the pre-baked fallback asset with
    status "fallback" and a diagnostic message.
    """
    prompt = build_image_prompt(scene_json, user_prompt=user_prompt)
    output_target = Path(output_path).with_suffix(".png")
    output_target.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[str] = []
    last_error = ""

    for provider_name in config.IMAGE_PROVIDER_ORDER:
        provider = get_provider(provider_name)
        if provider is None:
            continue

        for attempt in range(config.IMAGE_RETRY_COUNT + 1):
            try:
                started = time.time()
                raw = provider(prompt, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, seed)
                img = Image.open(io.BytesIO(raw))
                img.load()
                img = prepare_for_reconstruction(img)
                img.save(str(output_target), format="PNG")

                if not output_target.exists() or output_target.stat().st_size < 512:
                    raise RuntimeError("image file missing or truncated after save")

                elapsed = time.time() - started
                attempts.append(f"{provider_name}: ok in {elapsed:.1f}s")
                logger.info(
                    "Image generated via %s in %.1fs -> %s", provider_name, elapsed, output_target
                )
                return ImageResult(
                    image_path=str(output_target),
                    status="ok",
                    provider_used=provider_name,
                    attempts=attempts,
                    width=img.width,
                    height=img.height,
                )
            except Exception as exc:
                last_error = f"{provider_name}: {type(exc).__name__} - {str(exc)[:160]}"
                attempts.append(last_error)
                logger.warning("%s (attempt %d)", last_error, attempt + 1)
                if attempt < config.IMAGE_RETRY_COUNT:
                    time.sleep(1.5 * (attempt + 1))

    logger.warning("All image providers failed; using fallback asset.")
    return ImageResult(
        image_path=str(config.FALLBACK_IMAGE_PATH),
        status="fallback",
        provider_used=None,
        error_message=last_error or "No image provider configured.",
        attempts=attempts,
    )
