"""
GenFX blend worker - optional, for split deployments.

Some free hosts give you a front end with too little memory or too old a Python
to run Blender. Deploy this as its own Space, point the main app at it with
GENFX_BLEND_WORKER, and the heavy .blend build happens here instead.

    GENFX_BLEND_WORKER=https://<user>-genfx-worker.hf.space

Run locally with:  python worker/app.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.blend_builder import build_blend, describe_runtime  # noqa: E402
from app.mesh_gen import build_solid_from_image  # noqa: E402


def build_from_mesh(mesh_file, scene_json: str) -> tuple[str | None, str]:
    """Turn an uploaded mesh into a downloadable .blend."""
    if not mesh_file:
        return None, "No mesh uploaded."

    workdir = Path(tempfile.mkdtemp(prefix="genfx_worker_"))
    src = Path(mesh_file)
    local = workdir / src.name
    shutil.copy(src, local)

    scene = None
    if scene_json and scene_json.strip():
        try:
            scene = json.loads(scene_json)
        except json.JSONDecodeError as exc:
            return None, f"scene_json is not valid JSON: {exc}"

    result = build_blend(mesh_path=local, out_dir=workdir, scene_json=scene, make_preview=False)
    if result.status != "ok" or not result.blend_path:
        return None, f"Build failed: {result.error_message}"

    stats = result.stats or {}
    summary = (
        f"Built with {result.runtime} · "
        f"{stats.get('polygons', 0):,} polygons · "
        f"{stats.get('blend_bytes', 0) / 1e6:.1f} MB"
    )
    return result.blend_path, summary


def build_from_image(image_path, scene_json: str) -> tuple[str | None, str]:
    """Convenience path: image straight to .blend using the local mesh builder."""
    if not image_path:
        return None, "No image uploaded."

    workdir = Path(tempfile.mkdtemp(prefix="genfx_worker_img_"))
    try:
        obj, verts, faces = build_solid_from_image(str(image_path), workdir / "mesh")
    except Exception as exc:
        return None, f"Mesh build failed: {exc}"

    scene = None
    if scene_json and scene_json.strip():
        try:
            scene = json.loads(scene_json)
        except json.JSONDecodeError:
            scene = None

    result = build_blend(
        mesh_path=obj, out_dir=workdir, scene_json=scene,
        image_path=str(image_path), make_preview=False,
    )
    if result.status != "ok" or not result.blend_path:
        return None, f"Build failed: {result.error_message}"
    return result.blend_path, f"Built from image · {verts:,} vertices · {faces:,} faces"


runtime = describe_runtime()
runtime_note = (
    f"Blender binary: `{runtime.get('blender_binary') or 'none'}` · "
    f"bpy module: `{runtime.get('bpy_module')}`"
)

with gr.Blocks(title="GenFX blend worker", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        "# GenFX blend worker\n"
        "Converts a mesh into an editable `.blend` with materials, lighting and "
        "a framed camera. Point a GenFX front end at this Space with "
        "`GENFX_BLEND_WORKER`.\n\n" + runtime_note
    )

    with gr.Tab("Mesh → .blend"):
        mesh_in = gr.File(label="Mesh (.glb / .gltf / .obj / .ply / .stl)", type="filepath")
        scene_in = gr.Textbox(label="Scene JSON (optional)", lines=4)
        mesh_btn = gr.Button("Build .blend", variant="primary")
        mesh_out = gr.File(label="Result")
        mesh_msg = gr.Textbox(label="Status", interactive=False)
        mesh_btn.click(
            build_from_mesh, inputs=[mesh_in, scene_in],
            outputs=[mesh_out, mesh_msg], api_name="build_blend",
        )

    with gr.Tab("Image → .blend"):
        img_in = gr.Image(label="Reference image", type="filepath")
        img_scene = gr.Textbox(label="Scene JSON (optional)", lines=4)
        img_btn = gr.Button("Build .blend", variant="primary")
        img_out = gr.File(label="Result")
        img_msg = gr.Textbox(label="Status", interactive=False)
        img_btn.click(
            build_from_image, inputs=[img_in, img_scene],
            outputs=[img_out, img_msg], api_name="build_blend_from_image",
        )

if __name__ == "__main__":
    demo.queue(max_size=12).launch(server_name="0.0.0.0", server_port=7860)
