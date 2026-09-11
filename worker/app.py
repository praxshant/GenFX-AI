"""
GenFX blend worker - optional, for split deployments.

Some free hosts give you a front end with too little memory or too old a Python
to run Blender. Deploy this as its own Space, point the main app at it with
GENFX_BLEND_WORKER, and the heavy .blend build happens here instead.

    GENFX_BLEND_WORKER=https://<user>-genfx-worker.hf.space
    GENFX_WORKER_TOKEN=<same secret on both sides>

Run locally with:  python worker/app.py
"""

from __future__ import annotations

import hmac
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.blend_builder import build_blend, describe_runtime, extract_mesh_bundle  # noqa: E402
from app.mesh_gen import build_solid_from_image  # noqa: E402

WORK_ROOT = Path(tempfile.gettempdir()) / "genfx_worker"
KEEP_SECONDS = 3600

NO_FILES = (None, None, None)


def new_workdir(prefix: str) -> Path:
    """
    A scratch directory for one build, after sweeping stale ones.

    Gradio serves the .blend from wherever it was written, so the directory has
    to outlive the call. Without this sweep a long-lived Space accumulates every
    mesh it has ever built and eventually fills its disk.
    """
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - KEEP_SECONDS
    for stale in WORK_ROOT.iterdir():
        try:
            if stale.is_dir() and stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix=prefix, dir=WORK_ROOT))


def authorised(token: str | None) -> bool:
    """With GENFX_WORKER_TOKEN set, only callers that present it get a build."""
    if not config.BLEND_WORKER_TOKEN:
        return True
    return hmac.compare_digest((token or "").encode(), config.BLEND_WORKER_TOKEN.encode())


def _parse_scene(scene_json: str | None) -> dict | None:
    if not scene_json or not scene_json.strip():
        return None
    return json.loads(scene_json)


def _outputs(result) -> tuple[str | None, str | None, str | None]:
    return result.blend_path, result.glb_path, result.preview_path


def build_from_mesh(mesh_file, scene_json: str, image_file=None, make_preview: bool = False,
                    token: str = ""):
    """Turn an uploaded mesh (or a zipped OBJ + material + texture) into a .blend."""
    if not authorised(token):
        return (*NO_FILES, "Unauthorised: missing or wrong worker token.")
    if not mesh_file:
        return (*NO_FILES, "No mesh uploaded.")

    try:
        scene = _parse_scene(scene_json)
    except json.JSONDecodeError as exc:
        return (*NO_FILES, f"scene_json is not valid JSON: {exc}")

    workdir = new_workdir("mesh_")
    src = Path(mesh_file)
    try:
        if src.suffix.lower() == ".zip":
            local = extract_mesh_bundle(src, workdir / "mesh")
        else:
            local = workdir / src.name
            shutil.copy(src, local)
    except Exception as exc:
        return (*NO_FILES, f"Could not read the upload: {exc}")

    result = build_blend(
        mesh_path=local, out_dir=workdir, scene_json=scene,
        image_path=str(image_file) if image_file else None,
        make_preview=bool(make_preview),
    )
    if result.status != "ok" or not result.blend_path:
        return (*NO_FILES, f"Build failed: {result.error_message}")

    stats = result.stats or {}
    summary = (
        f"Built with {result.runtime} · "
        f"{stats.get('polygons', 0):,} polygons · "
        f"{stats.get('blend_bytes', 0) / 1e6:.1f} MB"
    )
    return (*_outputs(result), summary)


def build_from_image(image_path, scene_json: str, make_preview: bool = False, token: str = ""):
    """Convenience path: image straight to .blend using the local mesh builder."""
    if not authorised(token):
        return (*NO_FILES, "Unauthorised: missing or wrong worker token.")
    if not image_path:
        return (*NO_FILES, "No image uploaded.")

    try:
        scene = _parse_scene(scene_json)
    except json.JSONDecodeError as exc:
        return (*NO_FILES, f"scene_json is not valid JSON: {exc}")

    workdir = new_workdir("image_")
    try:
        obj, verts, faces = build_solid_from_image(str(image_path), workdir / "mesh")
    except Exception as exc:
        return (*NO_FILES, f"Mesh build failed: {exc}")

    result = build_blend(
        mesh_path=obj, out_dir=workdir, scene_json=scene,
        image_path=str(image_path), make_preview=bool(make_preview),
    )
    if result.status != "ok" or not result.blend_path:
        return (*NO_FILES, f"Build failed: {result.error_message}")
    return (*_outputs(result), f"Built from image · {verts:,} vertices · {faces:,} faces")


runtime = describe_runtime()
runtime_note = (
    f"Blender binary: `{runtime.get('blender_binary') or 'none'}` · "
    f"bpy module: `{runtime.get('bpy_module')}` · "
    f"token required: `{bool(config.BLEND_WORKER_TOKEN)}`"
)


def _result_widgets():
    return (
        gr.File(label=".blend"),
        gr.File(label="preview.glb"),
        gr.File(label="preview.png"),
        gr.Textbox(label="Status", interactive=False),
    )


with gr.Blocks(title="GenFX blend worker", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        "# GenFX blend worker\n"
        "Converts a mesh into an editable `.blend` with materials, lighting and "
        "a framed camera. Point a GenFX front end at this Space with "
        "`GENFX_BLEND_WORKER`.\n\n" + runtime_note
    )
    token_in = gr.Textbox(
        label="Worker token", type="password", visible=bool(config.BLEND_WORKER_TOKEN)
    )

    with gr.Tab("Mesh → .blend"):
        mesh_in = gr.File(label="Mesh (.glb / .gltf / .obj / .ply / .stl, or a zipped OBJ)",
                          type="filepath")
        scene_in = gr.Textbox(label="Scene JSON (optional)", lines=4)
        ref_in = gr.File(label="Reference image (optional)", type="filepath")
        prev_in = gr.Checkbox(label="Render preview", value=False)
        mesh_btn = gr.Button("Build .blend", variant="primary")
        mesh_outs = _result_widgets()
        mesh_btn.click(
            build_from_mesh, inputs=[mesh_in, scene_in, ref_in, prev_in, token_in],
            outputs=list(mesh_outs), api_name="build_blend",
        )

    with gr.Tab("Image → .blend"):
        img_in = gr.Image(label="Reference image", type="filepath")
        img_scene = gr.Textbox(label="Scene JSON (optional)", lines=4)
        img_prev = gr.Checkbox(label="Render preview", value=False)
        img_btn = gr.Button("Build .blend", variant="primary")
        img_outs = _result_widgets()
        img_btn.click(
            build_from_image, inputs=[img_in, img_scene, img_prev, token_in],
            outputs=list(img_outs), api_name="build_blend_from_image",
        )

if __name__ == "__main__":
    demo.queue(max_size=12).launch(
        server_name="0.0.0.0", server_port=7860, max_file_size="200mb"
    )
