"""
Central configuration for GenFX.

Every setting is env-overridable so the same image runs locally, on Hugging
Face Spaces, on Streamlit Community Cloud, or in a plain Docker host.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
BLENDER_DIR = PROJECT_ROOT / "blender"

# Runs live outside the source tree when RUNS_DIR is set (Spaces uses /data).
RUNS_DIR = Path(os.getenv("GENFX_RUNS_DIR", str(PROJECT_ROOT / "runs")))

FALLBACK_JSON_PATH = ASSETS_DIR / "fallback_scene.json"
FALLBACK_IMAGE_PATH = ASSETS_DIR / "fallback_image.png"
FALLBACK_RENDER_PATH = ASSETS_DIR / "fallback_render.png"

BLEND_BUILDER_SCRIPT = BLENDER_DIR / "build_blend.py"

# ── API keys ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HUGGINGFACE_API_KEY = (
    os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
).strip()

# ── Stage 1: prompt -> scene JSON ─────────────────────────────────────────────
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# Ollama: local, private, free. Never auto-installed - if the daemon is not
# running the provider simply reports unavailable and the cascade moves on.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_ENABLED = _env_bool("OLLAMA_ENABLED", True)

# Order matters: first available provider wins, and on failure we fall through.
LLM_PROVIDER_ORDER = [
    p.strip()
    for p in os.getenv("LLM_PROVIDER_ORDER", "openrouter,openai,huggingface,ollama").split(",")
    if p.strip()
]

# ── Stage 2: scene JSON -> image ──────────────────────────────────────────────
# Pollinations needs no API key at all, which is what keeps the free demo alive.
POLLINATIONS_ENDPOINT = "https://image.pollinations.ai/prompt"
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux")
POLLINATIONS_ENABLED = _env_bool("POLLINATIONS_ENABLED", True)

HF_IMAGE_MODELS = [
    m.strip()
    for m in os.getenv(
        "HF_IMAGE_MODELS",
        "black-forest-labs/FLUX.1-schnell,stabilityai/stable-diffusion-3.5-large-turbo",
    ).split(",")
    if m.strip()
]

IMAGE_PROVIDER_ORDER = [
    p.strip()
    for p in os.getenv("IMAGE_PROVIDER_ORDER", "pollinations,huggingface").split(",")
    if p.strip()
]

IMAGE_WIDTH = _env_int("GENFX_IMAGE_WIDTH", 1024)
IMAGE_HEIGHT = _env_int("GENFX_IMAGE_HEIGHT", 1024)

# ── Stage 3: image -> 3D mesh ─────────────────────────────────────────────────
# Each entry is a Hugging Face Space exposing a Gradio API we can call.
# "textured" spaces cost more ZeroGPU seconds, so they need a token; the
# shape-only space works anonymously and is the dependable default.
MESH_SPACES = [
    s.strip()
    for s in os.getenv(
        "GENFX_MESH_SPACES",
        "frogleo/Image-to-3D,trellis-community/TRELLIS,tencent/Hunyuan3D-2.1",
    ).split(",")
    if s.strip()
]

MESH_SPACE_TIMEOUT = _env_int("GENFX_MESH_TIMEOUT", 300)
MESH_TARGET_FACES = _env_int("GENFX_MESH_TARGET_FACES", 30000)
MESH_OCTREE_RESOLUTION = _env_int("GENFX_MESH_OCTREE", 256)
MESH_STEPS = _env_int("GENFX_MESH_STEPS", 30)

# Local depth-relief fallback (no GPU, no key, always available).
DEPTH_MODEL = os.getenv("GENFX_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DEPTH_LOCAL_ENABLED = _env_bool("GENFX_DEPTH_LOCAL", True)
RELIEF_GRID = _env_int("GENFX_RELIEF_GRID", 220)
RELIEF_DEPTH_SCALE = _env_float("GENFX_RELIEF_DEPTH", 0.45)

MESH_PROVIDER_ORDER = [
    p.strip()
    for p in os.getenv("MESH_PROVIDER_ORDER", "space,depth,relief").split(",")
    if p.strip()
]

# ── Stage 4: mesh -> .blend ───────────────────────────────────────────────────
# Either a real Blender binary or the pip-installable `bpy` module works.
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
BLEND_TIMEOUT_SECONDS = _env_int("GENFX_BLEND_TIMEOUT", 300)
BLEND_RENDER_PREVIEW = _env_bool("GENFX_BLEND_PREVIEW", True)
BLEND_PREVIEW_SAMPLES = _env_int("GENFX_PREVIEW_SAMPLES", 16)

# Remote worker: when set, .blend building is delegated to another deployment
# (a Gradio Space or any GenFX worker) instead of running bpy in this process.
BLEND_WORKER_URL = os.getenv("GENFX_BLEND_WORKER", "").strip()

# ── Retries & timeouts ────────────────────────────────────────────────────────
LLM_RETRY_COUNT = _env_int("GENFX_LLM_RETRIES", 1)
IMAGE_RETRY_COUNT = _env_int("GENFX_IMAGE_RETRIES", 2)
API_TIMEOUT_SECONDS = _env_int("GENFX_API_TIMEOUT", 90)
IMAGE_TIMEOUT_SECONDS = _env_int("GENFX_IMAGE_TIMEOUT", 120)

# ── Housekeeping ──────────────────────────────────────────────────────────────
MAX_RUNS_KEPT = _env_int("GENFX_MAX_RUNS", 40)
APP_VERSION = "2.0.0"


def summary() -> dict[str, object]:
    """Non-secret snapshot of the active configuration, for the UI and logs."""
    return {
        "version": APP_VERSION,
        "llm_order": LLM_PROVIDER_ORDER,
        "image_order": IMAGE_PROVIDER_ORDER,
        "mesh_order": MESH_PROVIDER_ORDER,
        "mesh_spaces": MESH_SPACES,
        "has_openrouter": bool(OPENROUTER_API_KEY),
        "has_openai": bool(OPENAI_API_KEY),
        "has_hf": bool(HUGGINGFACE_API_KEY),
        "ollama_enabled": OLLAMA_ENABLED,
        "blend_worker": bool(BLEND_WORKER_URL),
        "runs_dir": str(RUNS_DIR),
    }
