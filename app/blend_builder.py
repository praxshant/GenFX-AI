"""
Stage 4 - Mesh -> editable .blend.

The build always runs in a subprocess. Blender's Python can abort the whole
process on a bad import, and taking down the web app with it is not an option.

Runtimes, in order of preference:
  1. A real Blender executable (BLENDER_PATH or on PATH).
  2. The pip-installed `bpy` module in this interpreter - no Blender install
     needed, which is what makes free container deploys viable.
  3. A remote GenFX worker (GENFX_BLEND_WORKER), for split deployments where
     the front end is too small to host Blender.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app import config

logger = logging.getLogger(__name__)

RESULT_PREFIX = "GENFX_RESULT "


@dataclass
class BlendResult:
    blend_path: str | None
    status: str  # "ok" | "fallback"
    runtime: str  # "blender" | "bpy" | "worker" | "none"
    preview_path: str | None = None
    glb_path: str | None = None
    stats: dict = field(default_factory=dict)
    error_message: str | None = None
    log_path: str | None = None


# ── Runtime discovery ─────────────────────────────────────────────────────────

def resolve_blender_path(configured_path: str | None = None) -> str | None:
    """Find a Blender executable: explicit path, then PATH, then usual places."""
    configured_path = configured_path or config.BLENDER_PATH
    if configured_path:
        candidate = Path(configured_path)
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate)
        found = shutil.which(configured_path)
        if found:
            return found

    if sys.platform == "win32":
        base = Path(r"C:\Program Files\Blender Foundation")
        if base.is_dir():
            versions = sorted(base.glob("Blender */blender.exe"), reverse=True)
            if versions:
                return str(versions[0])
    elif sys.platform == "darwin":
        mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if mac.is_file():
            return str(mac)
    else:
        for path in ("/usr/bin/blender", "/usr/local/bin/blender", "/opt/blender/blender"):
            if Path(path).is_file():
                return path
    return None


def has_bpy_module() -> bool:
    """True when `import bpy` works in a fresh subprocess of this interpreter."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import bpy; print(bpy.app.version_string)"],
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0
    except Exception:
        return False


def describe_runtime() -> dict[str, object]:
    """What this deployment can actually do, for the health panel."""
    blender = resolve_blender_path()
    return {
        "blender_binary": blender,
        "bpy_module": has_bpy_module() if not blender else True,
        "worker": config.BLEND_WORKER_URL or None,
    }


# ── Build ─────────────────────────────────────────────────────────────────────

def _script_args(mesh_path: Path, output_path: Path, scene_json_path: Path | None,
                 image_path: str | None, preview_path: Path | None,
                 glb_path: Path | None) -> list[str]:
    args = ["--mesh", str(mesh_path), "--output", str(output_path)]
    if scene_json_path:
        args += ["--scene-json", str(scene_json_path)]
    if image_path and os.path.exists(image_path):
        args += ["--image", str(image_path)]
    if preview_path:
        args += ["--preview", str(preview_path)]
    if glb_path:
        args += ["--glb-out", str(glb_path)]
    args += ["--preview-samples", str(config.BLEND_PREVIEW_SAMPLES)]
    return args


def _parse_stats(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
    return {}


def _run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Blend build: %s", " ".join(cmd[:3]) + " ...")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=config.BLEND_TIMEOUT_SECONDS,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    combined = (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    return proc.returncode, proc.stdout or ""


def build_blend_via_worker(mesh_path: Path, out_dir: Path, scene_json_path: Path | None) -> BlendResult:
    """Delegate the build to a remote GenFX worker Space."""
    from gradio_client import handle_file

    from app.mesh_gen import make_gradio_client

    client = make_gradio_client(config.BLEND_WORKER_URL)
    scene_text = ""
    if scene_json_path and scene_json_path.exists():
        scene_text = scene_json_path.read_text(encoding="utf-8")

    result = client.predict(
        mesh_file=handle_file(str(mesh_path)),
        scene_json=scene_text,
        api_name="/build_blend",
    )

    blend_src = None
    for item in (result if isinstance(result, (list, tuple)) else [result]):
        candidate = item.get("value") if isinstance(item, dict) else item
        if isinstance(candidate, str) and candidate.lower().endswith(".blend"):
            blend_src = candidate
            break
    if not blend_src:
        raise RuntimeError(f"worker returned no .blend: {str(result)[:200]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "scene.blend"
    src = Path(blend_src)
    if src.exists():
        dest.write_bytes(src.read_bytes())
    else:
        import urllib.request

        url = blend_src if blend_src.startswith("http") else client.src.rstrip("/") + blend_src
        with urllib.request.urlopen(url, timeout=180) as resp:
            dest.write_bytes(resp.read())

    return BlendResult(blend_path=str(dest), status="ok", runtime="worker")


def build_blend(
    mesh_path: str | Path,
    out_dir: str | Path,
    scene_json: dict | None = None,
    image_path: str | None = None,
    make_preview: bool | None = None,
) -> BlendResult:
    """
    Build the .blend. Never raises; returns status "fallback" with a diagnostic
    when no runtime could produce a file.
    """
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        return BlendResult(
            blend_path=None, status="fallback", runtime="none",
            error_message=f"mesh not found: {mesh_path}",
        )

    output_path = out_dir / "scene.blend"
    log_path = out_dir / "blender.log"
    scene_json_path = None
    if scene_json is not None:
        scene_json_path = out_dir / "scene.json"
        scene_json_path.write_text(json.dumps(scene_json, indent=2), encoding="utf-8")

    make_preview = config.BLEND_RENDER_PREVIEW if make_preview is None else make_preview
    preview_path = (out_dir / "preview.png") if make_preview else None
    glb_path = out_dir / "preview.glb"

    script = str(config.BLEND_BUILDER_SCRIPT)
    script_args = _script_args(
        mesh_path, output_path, scene_json_path, image_path, preview_path, glb_path
    )

    errors: list[str] = []

    # 1) Real Blender executable
    blender_exe = resolve_blender_path()
    if blender_exe:
        try:
            code, stdout = _run(
                [blender_exe, "--background", "--factory-startup", "--python", script, "--", *script_args],
                log_path,
            )
            if output_path.exists() and output_path.stat().st_size > 1024:
                stats = _parse_stats(stdout)
                return BlendResult(
                    blend_path=str(output_path), status="ok", runtime="blender",
                    preview_path=str(preview_path) if preview_path and preview_path.exists() else None,
                    glb_path=str(glb_path) if glb_path.exists() else None,
                    stats=stats, log_path=str(log_path),
                )
            errors.append(f"blender exited {code} without writing the .blend")
        except subprocess.TimeoutExpired:
            errors.append(f"blender timed out after {config.BLEND_TIMEOUT_SECONDS}s")
        except Exception as exc:
            errors.append(f"blender: {type(exc).__name__} - {str(exc)[:160]}")

    # 2) pip `bpy` module
    try:
        code, stdout = _run([sys.executable, script, *script_args], log_path)
        if output_path.exists() and output_path.stat().st_size > 1024:
            stats = _parse_stats(stdout)
            return BlendResult(
                blend_path=str(output_path), status="ok", runtime="bpy",
                preview_path=str(preview_path) if preview_path and preview_path.exists() else None,
                glb_path=str(glb_path) if glb_path.exists() else None,
                stats=stats, log_path=str(log_path),
            )
        tail = log_path.read_text(encoding="utf-8")[-400:] if log_path.exists() else ""
        errors.append(f"bpy module exited {code}: {tail.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        errors.append(f"bpy module timed out after {config.BLEND_TIMEOUT_SECONDS}s")
    except Exception as exc:
        errors.append(f"bpy: {type(exc).__name__} - {str(exc)[:160]}")

    # 3) Remote worker
    if config.BLEND_WORKER_URL:
        try:
            return build_blend_via_worker(mesh_path, out_dir, scene_json_path)
        except Exception as exc:
            errors.append(f"worker: {type(exc).__name__} - {str(exc)[:160]}")

    return BlendResult(
        blend_path=None, status="fallback", runtime="none",
        error_message="; ".join(errors) or "no Blender runtime available",
        log_path=str(log_path) if log_path.exists() else None,
    )
