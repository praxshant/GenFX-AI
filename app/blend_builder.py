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
import zipfile
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


_BPY_AVAILABLE: bool | None = None


def has_bpy_module() -> bool:
    """
    True when `import bpy` works in a fresh subprocess of this interpreter.

    Cached: importing bpy costs seconds and hundreds of megabytes, and the
    health panel asks this on every page load. Only a definite answer is
    cached - a timeout on a cold container says nothing about bpy, and caching
    it would report "no Blender" for the life of the process.
    """
    global _BPY_AVAILABLE
    if _BPY_AVAILABLE is not None:
        return _BPY_AVAILABLE
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import bpy; print(bpy.app.version_string)"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return False
    _BPY_AVAILABLE = proc.returncode == 0
    return _BPY_AVAILABLE


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


# ── Remote worker ─────────────────────────────────────────────────────────────

MESH_FILE_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl", ".fbx")
BUNDLE_MAX_BYTES = 512 * 1024 * 1024


def bundle_mesh(mesh_path: Path, out_dir: Path) -> Path:
    """
    The file to upload to a worker for this mesh.

    A GLB carries its textures inside it and goes as it is. An OBJ does not:
    its material and texture sit beside it, and uploading the OBJ alone gets
    back an untextured .blend. Those go together as one zip.
    """
    if mesh_path.suffix.lower() != ".obj":
        return mesh_path
    bundle = out_dir / "mesh_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in mesh_path.parent.iterdir():
            if item.is_file() and item != bundle:
                zf.write(item, arcname=item.name)
    return bundle


def extract_mesh_bundle(bundle: Path, dest: Path) -> Path:
    """Unpack a bundle from bundle_mesh() and return the mesh inside it."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as zf:
        members = zf.infolist()
        if sum(m.file_size for m in members) > BUNDLE_MAX_BYTES:
            raise ValueError("mesh bundle is too large")
        for member in members:
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
                raise ValueError(f"unexpected path in mesh bundle: {member.filename}")
        zf.extractall(dest)
    meshes = sorted(
        (p for p in dest.iterdir() if p.suffix.lower() in MESH_FILE_EXTENSIONS),
        key=lambda p: MESH_FILE_EXTENSIONS.index(p.suffix.lower()),
    )
    if not meshes:
        raise ValueError("mesh bundle contains no mesh file")
    return meshes[0]


def _fetch_worker_file(item: str, base_url: str, dest: Path) -> None:
    """Gradio hands back either a local download or a path on the Space."""
    src = Path(item)
    if src.exists():
        dest.write_bytes(src.read_bytes())
        return
    import urllib.request

    url = item if item.startswith("http") else base_url.rstrip("/") + item
    with urllib.request.urlopen(url, timeout=180) as resp:
        dest.write_bytes(resp.read())


def build_blend_via_worker(
    mesh_path: Path,
    out_dir: Path,
    scene_json_path: Path | None,
    image_path: str | None = None,
    make_preview: bool = False,
) -> BlendResult:
    """Delegate the build to a remote GenFX worker Space."""
    from gradio_client import handle_file

    from app.mesh_gen import make_gradio_client

    client = make_gradio_client(config.BLEND_WORKER_URL, http_timeout=config.BLEND_TIMEOUT_SECONDS)
    scene_text = ""
    if scene_json_path and scene_json_path.exists():
        scene_text = scene_json_path.read_text(encoding="utf-8")

    upload = bundle_mesh(mesh_path, out_dir)
    result = client.predict(
        mesh_file=handle_file(str(upload)),
        scene_json=scene_text,
        image_file=handle_file(image_path) if image_path and os.path.exists(image_path) else None,
        make_preview=bool(make_preview),
        token=config.BLEND_WORKER_TOKEN,
        api_name="/build_blend",
    )

    items = result if isinstance(result, (list, tuple)) else [result]
    found: dict[str, str] = {}
    message = ""
    for item in items:
        candidate = item.get("value") if isinstance(item, dict) else item
        if not isinstance(candidate, str):
            continue
        low = candidate.lower().split("?")[0]
        for ext in (".blend", ".glb", ".png"):
            if low.endswith(ext):
                found.setdefault(ext, candidate)
                break
        else:
            message = candidate
    if ".blend" not in found:
        raise RuntimeError(f"worker returned no .blend: {message or str(result)[:200]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {".blend": out_dir / "scene.blend", ".glb": out_dir / "preview.glb",
               ".png": out_dir / "preview.png"}
    fetched: dict[str, str] = {}
    for ext, item in found.items():
        try:
            _fetch_worker_file(item, client.src, targets[ext])
            fetched[ext] = str(targets[ext])
        except Exception as exc:
            if ext == ".blend":
                raise
            logger.warning("Could not fetch worker %s: %s", ext, str(exc)[:140])

    return BlendResult(
        blend_path=fetched[".blend"], status="ok", runtime="worker",
        glb_path=fetched.get(".glb"), preview_path=fetched.get(".png"),
    )


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
    # Absolute from the start: Blender treats a relative path as relative to
    # the .blend it is writing, so a relative runs directory silently produces
    # an untextured file and a preview saved somewhere nobody looks.
    mesh_path = Path(mesh_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if image_path:
        image_path = str(Path(image_path).resolve())

    if not mesh_path.exists():
        return BlendResult(
            blend_path=None, status="fallback", runtime="none",
            error_message=f"mesh not found: {mesh_path}",
        )

    output_path = out_dir / "scene.blend"
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
    logs: list[Path] = []

    def attempt(runtime: str, cmd: list[str]) -> BlendResult | None:
        """Run one runtime. Returns a result on success, None to fall through."""
        # One log per runtime: a shared file means the second attempt erases the
        # evidence from the first, which is exactly the case worth diagnosing.
        log_path = out_dir / f"blender-{runtime}.log"
        logs.append(log_path)

        # A previous runtime may have left a truncated file behind; a stale one
        # would otherwise be mistaken for this attempt's output.
        if output_path.exists():
            output_path.unlink()

        try:
            code, stdout = _run(cmd, log_path)
        except subprocess.TimeoutExpired:
            errors.append(f"{runtime} timed out after {config.BLEND_TIMEOUT_SECONDS}s")
            return None
        except Exception as exc:
            errors.append(f"{runtime}: {type(exc).__name__} - {str(exc)[:160]}")
            return None

        if output_path.exists() and output_path.stat().st_size > 1024:
            return BlendResult(
                blend_path=str(output_path), status="ok", runtime=runtime,
                preview_path=str(preview_path) if preview_path and preview_path.exists() else None,
                glb_path=str(glb_path) if glb_path.exists() else None,
                stats=_parse_stats(stdout), log_path=str(log_path),
            )

        tail = log_path.read_text(encoding="utf-8")[-400:] if log_path.exists() else ""
        errors.append(f"{runtime} exited {code}: {tail.strip()[-200:] or 'no output'}")
        return None

    # 1) Real Blender executable
    blender_exe = resolve_blender_path()
    if blender_exe:
        result = attempt(
            "blender",
            [blender_exe, "--background", "--factory-startup", "--python", script, "--", *script_args],
        )
        if result:
            return result

    # 2) pip `bpy` module
    result = attempt("bpy", [sys.executable, script, *script_args])
    if result:
        return result

    # 3) Remote worker
    if config.BLEND_WORKER_URL:
        try:
            return build_blend_via_worker(
                mesh_path, out_dir, scene_json_path,
                image_path=image_path, make_preview=bool(preview_path),
            )
        except Exception as exc:
            errors.append(f"worker: {type(exc).__name__} - {str(exc)[:160]}")

    written = [p for p in logs if p.exists()]
    return BlendResult(
        blend_path=None, status="fallback", runtime="none",
        error_message="; ".join(errors) or "no Blender runtime available",
        log_path=str(written[-1]) if written else None,
    )
