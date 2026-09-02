"""
Builds the editable .blend file.

Runs in either Blender runtime:
    blender --background --python build_blend.py -- --mesh a.glb --output s.blend
    python build_blend.py --mesh a.glb --output s.blend        (pip `bpy` module)

What it produces is a scene a person can actually work in: the generated mesh
centred on the world origin and sitting on the ground, smooth-shaded, with a
real PBR material, a three-point light rig, a framed camera, and every texture
packed inside the file so the .blend is portable on its own.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


# ── Argument parsing (before bpy, so this file stays importable in tests) ─────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="GenFX .blend builder")
    parser.add_argument("--mesh", required=True, help="Input mesh (.glb/.gltf/.obj/.ply)")
    parser.add_argument("--output", required=True, help="Output .blend path")
    parser.add_argument("--scene-json", default=None, help="Scene JSON for look development")
    parser.add_argument("--image", default=None, help="Reference image to pack into the file")
    parser.add_argument("--preview", default=None, help="Optional preview PNG to render")
    parser.add_argument("--glb-out", default=None, help="Optional GLB export for web preview")
    parser.add_argument("--preview-samples", type=int, default=16)
    parser.add_argument("--no-ground", action="store_true", help="Skip the ground plane")
    return parser.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hex_to_rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """sRGB hex -> linear RGBA, which is what Blender node inputs expect."""
    try:
        h = (value or "").lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        srgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except Exception:
        srgb = [0.72, 0.72, 0.72]

    def to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (to_linear(c) for c in srgb)
    return (r, g, b, alpha)


def load_scene_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Blender scene construction ────────────────────────────────────────────────

def clear_scene(bpy) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for collection in (
        bpy.data.meshes, bpy.data.materials, bpy.data.images,
        bpy.data.lights, bpy.data.cameras, bpy.data.worlds,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_mesh(bpy, mesh_path: Path) -> list:
    """Import the mesh and return the objects that came in."""
    before = set(bpy.data.objects)
    suffix = mesh_path.suffix.lower()

    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == ".obj":
        # Our OBJ writer emits Z-up data, so import it without axis conversion.
        try:
            bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")
        except AttributeError:  # Blender < 3.3
            bpy.ops.import_scene.obj(filepath=str(mesh_path), axis_forward="Y", axis_up="Z")
    elif suffix == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        except AttributeError:
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    elif suffix == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=str(mesh_path))
        except AttributeError:
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")

    imported = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in {mesh_path.name}")
    return imported


def world_bounds(bpy, objects: list) -> tuple[list[float], list[float]]:
    import mathutils

    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                mins[i] = min(mins[i], world[i])
                maxs[i] = max(maxs[i], world[i])
    if mins[0] == float("inf"):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return mins, maxs


def normalise_subject(bpy, objects: list, target_size: float = 2.0) -> object:
    """
    Join the imported meshes, centre them on the origin, scale to a predictable
    size and drop them onto the ground plane. Returns the resulting object.
    """
    import mathutils

    meshes = [o for o in objects if o.type == "MESH"]
    for obj in bpy.data.objects:
        obj.select_set(False)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    # glTF arrives parented to a root empty that carries the Y-up to Z-up
    # conversion. Everything below reasons in world space, but `location` is
    # parent-relative - so the parent has to go first, or the centring lands in
    # the wrong place on exactly the highest-quality path.
    if any(o.parent for o in meshes):
        try:
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        except Exception as exc:
            print(f"WARN: parent_clear failed: {exc}", file=sys.stderr)

    if len(meshes) > 1:
        bpy.ops.object.join()
    subject = bpy.context.view_layer.objects.active
    subject.name = "GenFX_Subject"
    if subject.data:
        subject.data.name = "GenFX_SubjectMesh"

    # Bake any import rotation/scale into the mesh data.
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    mins, maxs = world_bounds(bpy, [subject])
    size = max(maxs[i] - mins[i] for i in range(3))
    if size <= 1e-6:
        size = 1.0
    scale = target_size / size

    centre = mathutils.Vector(
        ((mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, (mins[2] + maxs[2]) / 2)
    )
    subject.location -= centre
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    subject.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Rest the object on Z=0 so the ground plane reads correctly.
    mins, maxs = world_bounds(bpy, [subject])
    subject.location.z -= mins[2]
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # Smooth the curved surfaces but keep the silhouette crisp. Blender 4.1
    # replaced mesh auto-smooth with this operator; on anything older, fall
    # back to plain smooth shading, which melts hard edges but still beats the
    # faceted look of a raw grid.
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(50))
    except Exception:
        bpy.ops.object.shade_smooth()
        try:
            bpy.ops.object.modifier_add(type="WEIGHTED_NORMAL")
            subject.modifiers[-1].keep_sharp = True
        except Exception:
            pass

    return subject


def has_image_texture(subject) -> bool:
    for slot in subject.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                return True
    return False


def build_material(bpy, scene_json: dict, name: str = "GenFX_Surface"):
    """A clean Principled BSDF driven by the scene JSON."""
    materials = scene_json.get("materials", {}) if isinstance(scene_json, dict) else {}
    base_color = hex_to_rgba(materials.get("base_color", "#b8b8b8"))
    roughness = float(materials.get("roughness", 0.45) or 0.45)
    metallic = float(materials.get("metallic", 0.0) or 0.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = max(0.02, min(1.0, roughness))
        bsdf.inputs["Metallic"].default_value = max(0.0, min(1.0, metallic))
    # Viewport display colour, so solid/Workbench shading matches the render.
    mat.diffuse_color = base_color
    mat.roughness = max(0.02, min(1.0, roughness))
    mat.metallic = max(0.0, min(1.0, metallic))
    return mat


LIGHT_PRESETS = {
    "three_point_studio": {"key": 400.0, "fill": 120.0, "rim": 250.0, "world": 0.35},
    "cinematic_sunset":   {"key": 550.0, "fill":  70.0, "rim": 420.0, "world": 0.22},
    "soft_daylight":      {"key": 300.0, "fill": 200.0, "rim": 160.0, "world": 0.70},
    "dramatic_low_key":   {"key": 600.0, "fill":  25.0, "rim": 380.0, "world": 0.08},
    "neutral_product":    {"key": 350.0, "fill": 180.0, "rim": 200.0, "world": 0.50},
}


def build_lighting(bpy, scene_json: dict, radius: float = 4.0) -> None:
    lighting = scene_json.get("lighting", {}) if isinstance(scene_json, dict) else {}
    preset = LIGHT_PRESETS.get(
        str(lighting.get("preset", "")).lower(), LIGHT_PRESETS["three_point_studio"]
    )
    intensity = float(lighting.get("intensity", 1.0) or 1.0)
    key_color = hex_to_rgba(lighting.get("key_color", "#ffffff"))[:3]

    world = bpy.data.worlds.new("GenFX_World")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.055, 0.065, 1.0)
        bg.inputs["Strength"].default_value = preset["world"] * intensity
    bpy.context.scene.world = world

    def add_area(name, location, rotation, energy, size, color):
        data = bpy.data.lights.new(name=name, type="AREA")
        data.energy = energy
        data.size = size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        obj.location = location
        obj.rotation_euler = rotation
        bpy.context.collection.objects.link(obj)
        return obj

    r = radius
    add_area("GenFX_Key", (r * 0.9, -r * 0.9, r * 1.0),
             (math.radians(52), 0.0, math.radians(45)),
             preset["key"] * intensity, r * 1.1, key_color)
    add_area("GenFX_Fill", (-r * 1.1, -r * 0.6, r * 0.45),
             (math.radians(72), 0.0, math.radians(-58)),
             preset["fill"] * intensity, r * 1.6, (0.85, 0.90, 1.0))
    add_area("GenFX_Rim", (-r * 0.3, r * 1.2, r * 0.9),
             (math.radians(122), 0.0, math.radians(-18)),
             preset["rim"] * intensity, r * 0.8, (1.0, 0.96, 0.90))


SHOT_ELEVATION = {
    "low": 8.0, "low_angle": 8.0, "worm": 4.0,
    "eye_level": 20.0, "eye": 20.0, "neutral": 20.0,
    "high": 38.0, "high_angle": 38.0, "top": 62.0, "birds_eye": 62.0,
}


def build_camera(bpy, scene_json: dict, subject) -> object:
    import mathutils

    cam_cfg = scene_json.get("camera", {}) if isinstance(scene_json, dict) else {}
    focal = float(cam_cfg.get("focal_length", 50) or 50)
    elevation = SHOT_ELEVATION.get(str(cam_cfg.get("angle", "eye_level")).lower(), 20.0)

    shot = str(cam_cfg.get("shot_type", "three_quarter")).lower()
    margin = {"close_up": 1.02, "medium": 1.12, "three_quarter": 1.18, "wide": 1.6}.get(shot, 1.18)

    data = bpy.data.cameras.new("GenFX_Camera")
    data.lens = focal
    data.clip_start = 0.01
    data.clip_end = 500.0
    cam = bpy.data.objects.new("GenFX_Camera", data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    mins, maxs = world_bounds(bpy, [subject])
    centre = mathutils.Vector(
        ((mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, (mins[2] + maxs[2]) / 2)
    )
    dims = [maxs[i] - mins[i] for i in range(3)]
    extent = max(max(dims), 1e-3)

    # Blender's AUTO sensor fit maps sensor_width to the longer image axis.
    scene = bpy.context.scene
    res_x = max(1, scene.render.resolution_x)
    res_y = max(1, scene.render.resolution_y)
    sensor = data.sensor_width or 36.0
    tan_major = (sensor / 2.0) / focal
    minor_ratio = (res_y / res_x) if res_x >= res_y else (res_x / res_y)
    tan_minor = tan_major * minor_ratio
    if res_x >= res_y:
        tan_h, tan_v = tan_major, tan_minor
    else:
        tan_h, tan_v = tan_minor, tan_major
    # Margin as headroom around the subject rather than extra distance.
    tan_h = max(tan_h / margin, 1e-3)
    tan_v = max(tan_v / margin, 1e-3)

    azimuth = math.radians(-35.0)
    elev = math.radians(elevation)
    offset = mathutils.Vector(
        (math.cos(elev) * math.sin(azimuth), -math.cos(elev) * math.cos(azimuth), math.sin(elev))
    )

    # Fit the eight bounding-box corners, not the bounding sphere around them.
    # A sphere is the box's diagonal - for a 2m subject that is 3.4m of framing,
    # and the asset ends up half the height of its own preview.
    forward = -offset
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up)
    right = right.normalized() if right.length > 1e-6 else mathutils.Vector((1.0, 0.0, 0.0))
    up = right.cross(forward).normalized()

    distance = extent * 0.75
    for cx in (mins[0], maxs[0]):
        for cy in (mins[1], maxs[1]):
            for cz in (mins[2], maxs[2]):
                v = mathutils.Vector((cx, cy, cz)) - centre
                depth = v.dot(forward)
                distance = max(
                    distance,
                    abs(v.dot(right)) / tan_h - depth,
                    abs(v.dot(up)) / tan_v - depth,
                )

    cam.location = centre + offset * distance

    direction = centre - mathutils.Vector(cam.location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # A track-to constraint keeps the camera aimed if the artist moves it.
    empty = bpy.data.objects.new("GenFX_CameraTarget", None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = extent * 0.15
    empty.location = centre
    bpy.context.collection.objects.link(empty)

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = empty
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    cam.rotation_euler = (0.0, 0.0, 0.0)

    return cam


def build_ground(bpy, subject) -> object:
    mins, maxs = world_bounds(bpy, [subject])
    extent = max(max(maxs[i] - mins[i] for i in range(3)), 1e-3)

    # A hair below zero. The subject rests exactly on z=0, and a mesh with a
    # flat underside - which is most of what the hosted 3D models return -
    # would otherwise be coplanar with the plane and z-fight across its base.
    bpy.ops.mesh.primitive_plane_add(size=extent * 14.0, location=(0, 0, -extent * 1e-3))
    ground = bpy.context.active_object
    ground.name = "GenFX_Ground"

    mat = bpy.data.materials.new("GenFX_Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.16, 0.16, 0.17, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
    ground.data.materials.append(mat)
    return ground


def configure_render(bpy, scene_json: dict, samples: int) -> None:
    scene = bpy.context.scene
    rs = scene_json.get("render_settings", {}) if isinstance(scene_json, dict) else {}

    resolution = str(rs.get("resolution", "1920x1080"))
    try:
        w, h = (int(v) for v in resolution.lower().split("x")[:2])
    except Exception:
        w, h = 1920, 1080
    scene.render.resolution_x = max(64, min(7680, w))
    scene.render.resolution_y = max(64, min(4320, h))
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in engines:
            scene.render.engine = candidate
            break
    else:
        scene.render.engine = "CYCLES"

    try:
        scene.eevee.taa_render_samples = max(4, samples)
    except Exception:
        pass
    try:
        scene.cycles.samples = max(4, int(rs.get("samples", 64)))
        scene.cycles.use_denoising = True
    except Exception:
        pass

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def render_preview(bpy, output_path: str, samples: int) -> bool:
    """Fast Workbench turntable frame - seconds on CPU, unlike a Cycles pass."""
    scene = bpy.context.scene
    prev_engine = scene.render.engine
    prev_x, prev_y = scene.render.resolution_x, scene.render.resolution_y
    prev_path = scene.render.filepath
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        # Keep the scene's aspect ratio, or the framing the camera was built
        # for stops matching what the preview shows.
        aspect = (prev_x / prev_y) if prev_y else 1.0
        if aspect >= 1.0:
            scene.render.resolution_x = 900
            scene.render.resolution_y = max(64, int(round(900 / aspect)))
        else:
            scene.render.resolution_y = 900
            scene.render.resolution_x = max(64, int(round(900 * aspect)))
        scene.render.filepath = output_path
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        shading.show_shadows = True
        shading.show_cavity = True
        scene.display.render_aa = "8"
        bpy.ops.render.render(write_still=True)
        return os.path.exists(output_path)
    except Exception as exc:
        print(f"WARN: preview render failed: {exc}", file=sys.stderr)
        return False
    finally:
        scene.render.engine = prev_engine
        scene.render.resolution_x, scene.render.resolution_y = prev_x, prev_y
        scene.render.filepath = prev_path


def export_glb(bpy, path: str, subject=None) -> bool:
    """
    Export the subject alone for the web turntable.

    Exporting the whole scene would drag the ground plane in with it, and that
    plane is fourteen times the subject's extent - a viewer that auto-frames
    the model then shows a huge floor with a speck in the middle. The .blend
    keeps the full set dressing; the preview only needs the asset.
    """
    try:
        if subject is not None:
            for obj in bpy.data.objects:
                obj.select_set(obj is subject)
            bpy.context.view_layer.objects.active = subject

        bpy.ops.export_scene.gltf(
            filepath=path,
            export_format="GLB",
            use_selection=subject is not None,
            export_apply=True,
        )
        return os.path.exists(path)
    except Exception as exc:
        print(f"WARN: GLB export failed: {exc}", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Blender resolves a relative path against the .blend file's own directory,
    # not the working directory - so a relative --output lands somewhere else
    # entirely, the texture next to the OBJ is never found, and pack_all has
    # nothing to pack. Absolute paths from here on, whatever the caller passed.
    args.mesh = os.path.abspath(args.mesh)
    args.output = os.path.abspath(args.output)
    for name in ("scene_json", "image", "preview", "glb_out"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, os.path.abspath(value))

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"ERROR: mesh not found: {mesh_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import bpy  # noqa: E402  - deliberately deferred

    scene_json = load_scene_json(args.scene_json)
    print(f"GenFX blend builder | Blender {bpy.app.version_string} | mesh={mesh_path.name}")

    clear_scene(bpy)
    imported = import_mesh(bpy, mesh_path)
    subject = normalise_subject(bpy, imported)

    if not has_image_texture(subject):
        subject.data.materials.clear()
        subject.data.materials.append(build_material(bpy, scene_json))
        print("Applied generated PBR material (mesh had no texture).")
    else:
        print("Kept the imported textures.")

    build_lighting(bpy, scene_json)
    if not args.no_ground:
        build_ground(bpy, subject)
    # Resolution first: camera framing depends on the render aspect ratio.
    configure_render(bpy, scene_json, args.preview_samples)
    build_camera(bpy, scene_json, subject)

    # Keep the source image inside the file so the artist has the reference.
    if args.image and os.path.exists(args.image):
        try:
            ref = bpy.data.images.load(args.image)
            ref.name = "GenFX_Reference"
            ref.use_fake_user = True
        except Exception as exc:
            print(f"WARN: could not load reference image: {exc}", file=sys.stderr)

    # Pack every external file so the .blend opens correctly anywhere.
    try:
        bpy.ops.file.pack_all()
    except Exception as exc:
        print(f"WARN: pack_all failed: {exc}", file=sys.stderr)

    for obj in bpy.data.objects:
        obj.select_set(obj is subject)
    bpy.context.view_layer.objects.active = subject

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=True)
    if not output_path.exists() or output_path.stat().st_size < 1024:
        print("ERROR: .blend was not written", file=sys.stderr)
        return 3

    stats = {
        "blend": str(output_path),
        "blend_bytes": output_path.stat().st_size,
        "blender_version": bpy.app.version_string,
        "vertices": len(subject.data.vertices),
        "polygons": len(subject.data.polygons),
        "materials": [m.name for m in subject.data.materials if m],
    }

    # Both of these run after the save, so neither can corrupt the deliverable.
    if args.glb_out:
        stats["glb"] = args.glb_out if export_glb(bpy, args.glb_out, subject) else None
    if args.preview:
        stats["preview"] = args.preview if render_preview(bpy, args.preview, args.preview_samples) else None

    print("GENFX_RESULT " + json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
