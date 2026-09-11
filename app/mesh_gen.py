"""
Stage 3 - Reference image -> 3D mesh.

Three tiers, tried in order, so there is always a mesh:

  1. "space"  - a real image-to-3D model (Hunyuan3D / TRELLIS) called through a
                public Hugging Face Space's Gradio API. Free; an HF token buys
                more ZeroGPU quota and unlocks the textured pipelines.
  2. "depth"  - monocular depth estimation, then a displaced + inflated mesh.
                Runs locally when the optional transformers + torch are
                installed; textured with the source image.
  3. "relief" - no model at all: the subject is segmented from its background
                and inflated with a distance transform. Pure numpy, always
                works, always textured.

Tiers 2 and 3 emit OBJ + MTL + texture; tier 1 emits GLB (or OBJ). The blend
builder accepts either.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app import config

logger = logging.getLogger(__name__)

MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl")

# How far, in grid cells, the rim's UVs are pulled inside the silhouette.
RIM_UV_INSET = 3


@dataclass
class MeshResult:
    mesh_path: str | None
    status: str  # "ok" | "fallback"
    method: str  # "space" | "depth" | "relief" | "none"
    provider_used: str | None = None
    textured: bool = False
    error_message: str | None = None
    attempts: list[str] = field(default_factory=list)
    vertex_count: int = 0
    face_count: int = 0
    elapsed: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 - Hugging Face Space adapters
# ══════════════════════════════════════════════════════════════════════════════

def _space_payload(space_id: str, image_path: str, handle_file) -> tuple[str, dict]:
    """Return (api_name, kwargs) for a known Space."""
    sid = space_id.lower()

    if "trellis" in sid:
        return "/generate_and_extract_glb", {
            "image": handle_file(image_path),
            "multiimages": [],
            "seed": 0,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 12,
            "slat_guidance_strength": 3.0,
            "slat_sampling_steps": 12,
            "multiimage_algo": "stochastic",
            "mesh_simplify": 0.95,
            "texture_size": 1024,
        }

    if "hunyuan3d" in sid:
        kwargs = {
            "image": handle_file(image_path),
            "mv_image_front": None,
            "mv_image_back": None,
            "mv_image_left": None,
            "mv_image_right": None,
            "steps": config.MESH_STEPS,
            "guidance_scale": 5.0,
            "seed": 1234,
            "octree_resolution": config.MESH_OCTREE_RESOLUTION,
            "check_box_rembg": True,
            "num_chunks": 8000,
            "randomize_seed": False,
        }
        if space_id == "tencent/Hunyuan3D-2":
            kwargs["caption"] = ""
        return "/shape_generation", kwargs

    # frogleo/Image-to-3D and API-compatible forks
    return "/gen_shape", {
        "image": handle_file(image_path),
        "steps": config.MESH_STEPS,
        "guidance_scale": 5.0,
        "seed": 1234,
        "octree_resolution": config.MESH_OCTREE_RESOLUTION,
        "num_chunks": 8000,
        "target_face_num": config.MESH_TARGET_FACES,
        "randomize_seed": False,
    }


def _iter_result_items(result):
    """Flatten a Gradio result of unknown shape into candidate scalars."""
    if result is None:
        return
    if isinstance(result, (str, bytes)):
        yield result
        return
    if isinstance(result, dict):
        for key in ("value", "path", "url", "name"):
            if key in result:
                yield from _iter_result_items(result[key])
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            yield from _iter_result_items(item)
        return
    yield result


def _harvest_mesh_file(result, base_url: str, dest_dir: Path) -> tuple[Path, bool] | None:
    """
    Find a mesh in a Gradio result and put it in dest_dir.

    Handles both shapes Spaces return: a local path (gradio_client already
    downloaded it) and a URL path on the Space that we must fetch ourselves.
    Prefers textured meshes (GLB) over shape-only ones.
    """
    candidates: list[tuple[int, str]] = []
    for item in _iter_result_items(result):
        if not isinstance(item, str):
            continue
        low = item.lower().split("?")[0]
        if not low.endswith(MESH_EXTENSIONS):
            continue
        # Rank: textured glb > gltf > obj > everything else
        rank = 0
        if low.endswith((".glb", ".gltf")):
            rank = 3
        elif low.endswith(".obj"):
            rank = 2
        else:
            rank = 1
        if "white_mesh" in low or "shape" in low:
            rank -= 1  # untextured hint
        candidates.append((rank, item))

    if not candidates:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    for _, item in sorted(candidates, key=lambda c: -c[0]):
        try:
            suffix = Path(item.split("?")[0]).suffix or ".glb"
            dest = dest_dir / f"mesh{suffix}"

            local = Path(item)
            if local.exists() and local.is_file():
                dest.write_bytes(local.read_bytes())
            else:
                url = item if item.startswith("http") else base_url.rstrip("/") + item
                req = urllib.request.Request(url, headers={"User-Agent": "GenFX/2.0"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    dest.write_bytes(resp.read())

            if dest.exists() and dest.stat().st_size > 256:
                textured = "white_mesh" not in item.lower()
                return dest, textured
        except Exception as exc:
            logger.warning("Could not retrieve %s: %s", item, str(exc)[:140])
    return None


def make_gradio_client(src: str, token: str | None = None, http_timeout: float = 60.0):
    """
    Build a gradio_client.Client across versions: the token kwarg was renamed
    from `hf_token` to `token`, `httpx_kwargs` is missing from old releases,
    and a bad kwarg is a hard TypeError.
    """
    from gradio_client import Client

    token = token or config.HUGGINGFACE_API_KEY or None
    for auth in ({"token": token}, {"hf_token": token}, {}):
        for extra in ({"httpx_kwargs": {"timeout": http_timeout}}, {}):
            try:
                return Client(src, verbose=False, **auth, **extra)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
    return Client(src, verbose=False)


def _call_with_deadline(client, deadline: float, api_name: str, **kwargs):
    """
    Submit a Space job and wait no longer than the tier's remaining budget.

    `predict()` blocks for as long as the Space's queue does, which on an
    anonymous ZeroGPU quota can be indefinitely - and the page waits with it.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("mesh time budget exhausted")
    job = client.submit(api_name=api_name, **kwargs)
    try:
        return job.result(timeout=remaining)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def generate_mesh_via_space(image_path: str, out_dir: Path) -> tuple[Path, str, bool]:
    """
    Try each configured Space until one returns a mesh, within a total budget
    of MESH_SPACE_TIMEOUT seconds shared by all of them.
    Returns (mesh_path, space_id, textured). Raises on total failure.
    """
    from gradio_client import handle_file

    errors: list[str] = []
    deadline = time.time() + config.MESH_SPACE_TIMEOUT

    for space_id in config.MESH_SPACES:
        if time.time() >= deadline:
            errors.append(f"{space_id}: skipped, {config.MESH_SPACE_TIMEOUT}s budget exhausted")
            continue
        try:
            logger.info("Requesting mesh from Space %s", space_id)
            client = make_gradio_client(space_id)

            try:  # TRELLIS-style session-scoped Spaces
                _call_with_deadline(client, min(deadline, time.time() + 20), "/start_session")
            except Exception:
                pass

            api_name, kwargs = _space_payload(space_id, image_path, handle_file)
            result = _call_with_deadline(client, deadline, api_name, **kwargs)

            found = _harvest_mesh_file(result, client.src, out_dir)
            if found is None:
                raise RuntimeError(f"no mesh file in response: {str(result)[:180]}")

            mesh_path, textured = found
            logger.info("Mesh from %s -> %s (%d bytes)", space_id, mesh_path, mesh_path.stat().st_size)
            return mesh_path, space_id, textured

        except Exception as exc:
            msg = f"{space_id}: {type(exc).__name__} - {str(exc)[:200]}"
            errors.append(msg)
            logger.warning("Space failed - %s", msg)

    raise RuntimeError("; ".join(errors) or "no Spaces configured")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2/3 - local mesh construction
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _depth_pipeline():
    """
    The transformers depth model, loaded once per process. Building it per run
    re-reads the weights from disk every time; the first build downloads them.
    Returns None when transformers/torch are not installed - they are optional.
    """
    try:
        from transformers import pipeline  # type: ignore

        return pipeline("depth-estimation", model=config.DEPTH_MODEL, device=-1)
    except Exception as exc:
        logger.info("Local depth model unavailable: %s", str(exc)[:140])
        return None


def _estimate_depth_local(img: Image.Image) -> np.ndarray | None:
    """Monocular depth via transformers, if torch happens to be installed."""
    if not config.DEPTH_LOCAL_ENABLED:
        return None
    pipe = _depth_pipeline()
    if pipe is None:
        return None
    try:
        depth = np.array(pipe(img)["depth"], dtype=np.float32)
        rng = depth.max() - depth.min()
        return (depth - depth.min()) / rng if rng > 1e-6 else None
    except Exception as exc:
        logger.info("Local depth estimation failed: %s", str(exc)[:140])
        return None


def background_model(arr: np.ndarray, band: int = 8) -> np.ndarray:
    """
    Estimate the backdrop colour at *every* pixel, not just one colour overall.

    A single median works for a flat background and fails for the studio
    gradients text-to-image models actually produce - dark above, warm below.
    Against one colour, half the backdrop reads as subject and the mesh comes
    out a slab. Interpolating the four border strips tracks a smooth gradient
    in either direction, and costs one pass over the image.
    """
    h, w, _ = arr.shape
    band = max(1, min(band, h // 4, w // 4))
    f = arr.astype(np.float32)

    left = np.median(f[:, :band], axis=1)      # (h, 3) - backdrop down the left
    right = np.median(f[:, -band:], axis=1)    # (h, 3)
    top = np.median(f[:band, :], axis=0)       # (w, 3) - backdrop across the top
    bottom = np.median(f[-band:, :], axis=0)   # (w, 3)

    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]

    horizontal = left[:, None, :] * (1.0 - u) + right[:, None, :] * u
    vertical = top[None, :, :] * (1.0 - v) + bottom[None, :, :] * v
    return 0.5 * (horizontal + vertical)


def foreground_mask(arr: np.ndarray) -> np.ndarray:
    """
    Separate subject from background using the borders as the reference.
    Works well because stage 2 asks for a single centred subject on purpose.
    """
    h, w, _ = arr.shape
    dist = np.linalg.norm(arr.astype(np.float32) - background_model(arr), axis=2)

    spread = float(np.percentile(dist, 98))
    threshold = max(18.0, spread * 0.22)
    mask = dist > threshold

    # Keep the blob the subject is in; drop specks and any backdrop that slipped
    # through. Stage 2 asks for one centred subject, so the centre is the seed.
    mask = _subject_component(mask)
    mask = _binary_close(mask, 2)

    if mask.mean() < 0.02:  # segmentation failed - treat the frame as the subject
        mask = np.ones((h, w), dtype=bool)
    return mask


def _subject_component(mask: np.ndarray) -> np.ndarray:
    """
    Flood-fill without scipy and keep one blob: the one covering the centre if
    there is one, else the largest.

    Preferring the centre matters when part of the backdrop survives
    thresholding - it is often larger than the subject, and picking by size
    alone hands back the wall instead of the object standing in front of it.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    best_label, best_size = 0, 0
    stack: list[tuple[int, int]] = []

    for sy in range(0, h, 2):
        for sx in range(0, w, 2):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            size = 0
            stack.append((sy, sx))
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            if size > best_size:
                best_size, best_label = size, current

    if not best_label:
        return mask

    # A subject that is centred but smaller than a surviving patch of backdrop
    # should still win. Look in a small window around the middle of the frame.
    centre = labels[
        max(0, h // 2 - h // 12): h // 2 + h // 12 + 1,
        max(0, w // 2 - w // 12): w // 2 + w // 12 + 1,
    ]
    seeds = centre[centre > 0]
    if seeds.size:
        values, counts = np.unique(seeds, return_counts=True)
        return labels == int(values[counts.argmax()])

    return labels == best_label


def _binary_close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate then erode, to seal pinholes in the silhouette."""
    def shift_or(m):
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        return out

    def shift_and(m):
        out = m.copy()
        out[1:, :] &= m[:-1, :]
        out[:-1, :] &= m[1:, :]
        out[:, 1:] &= m[:, :-1]
        out[:, :-1] &= m[:, 1:]
        return out

    for _ in range(iterations):
        mask = shift_or(mask)
    for _ in range(iterations):
        mask = shift_and(mask)
    return mask


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Two-pass chamfer distance to the nearest background pixel.
    Feeding this through a sqrt gives a natural dome, which is what turns a
    silhouette into a believable solid rather than a flat cut-out.
    """
    h, w = mask.shape
    INF = 1e9
    dist = np.where(mask, INF, 0.0).astype(np.float32)

    for y in range(h):
        row = dist[y]
        if y > 0:
            prev = dist[y - 1]
            np.minimum(row, prev + 1.0, out=row)
            np.minimum(row[1:], prev[:-1] + 1.4142, out=row[1:])
            np.minimum(row[:-1], prev[1:] + 1.4142, out=row[:-1])
        for x in range(1, w):
            if row[x] > row[x - 1] + 1.0:
                row[x] = row[x - 1] + 1.0

    for y in range(h - 1, -1, -1):
        row = dist[y]
        if y < h - 1:
            nxt = dist[y + 1]
            np.minimum(row, nxt + 1.0, out=row)
            np.minimum(row[1:], nxt[:-1] + 1.4142, out=row[1:])
            np.minimum(row[:-1], nxt[1:] + 1.4142, out=row[:-1])
        for x in range(w - 2, -1, -1):
            if row[x] > row[x + 1] + 1.0:
                row[x] = row[x + 1] + 1.0

    return dist


def _break_diagonal_pinches(cell: np.ndarray, max_passes: int = 12) -> np.ndarray:
    """
    Drop cells that meet a neighbour only at a corner.

    Two quads touching diagonally share a single grid point and no edge. The rim
    then runs through that point twice, which makes it a non-manifold vertex:
    Blender still imports it, but solidify, boolean, remesh and 3D printing all
    misbehave there. Clearing one of each diagonal pair keeps the surface a
    clean 2-manifold, at the cost of one grid cell.
    """
    cell = cell.copy()
    for _ in range(max_passes):
        backslash = cell[:-1, :-1] & cell[1:, 1:] & ~cell[:-1, 1:] & ~cell[1:, :-1]
        slash = cell[:-1, 1:] & cell[1:, :-1] & ~cell[:-1, :-1] & ~cell[1:, 1:]
        if not (backslash.any() or slash.any()):
            break
        cell[1:, 1:] &= ~backslash
        cell[1:, :-1] &= ~slash
    return cell


def build_solid_from_image(
    image_path: str,
    out_dir: Path,
    grid: int | None = None,
    depth_scale: float | None = None,
    depth_map: np.ndarray | None = None,
) -> tuple[Path, int, int]:
    """
    Build a closed, textured solid from a single image.

    Front surface = inflation dome (optionally modulated by a real depth map),
    back surface = shallower mirror, joined by a rim along the silhouette.
    Writes OBJ + MTL + texture PNG and returns (obj_path, verts, faces).

    The result is watertight and consistently wound outward, which is what lets
    Blender smooth-shade and light it correctly, and what lets a glTF viewer
    show it with backface culling on. Both properties are asserted in the tests
    (every edge used by exactly two faces; positive signed volume).
    """
    grid = grid or config.RELIEF_GRID
    depth_scale = depth_scale if depth_scale is not None else config.RELIEF_DEPTH_SCALE

    src = Image.open(image_path).convert("RGB")
    work = src.resize((grid, grid), Image.LANCZOS)
    arr = np.asarray(work)

    mask = foreground_mask(arr)

    dist = distance_transform(mask)
    if dist.max() > 0:
        dome = np.sqrt(dist / dist.max())
    else:
        dome = np.zeros_like(dist)

    if depth_map is not None:
        dm = np.asarray(
            Image.fromarray((depth_map * 255).astype(np.uint8)).resize((grid, grid), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        # Depth model supplies the silhouette detail, the dome supplies volume.
        height = dome * (0.45 + 0.55 * dm)
    else:
        height = dome

    height = height * mask

    # A quad exists only where all four of its grid corners are inside the
    # silhouette. Deriving the surface from cells rather than points is what
    # keeps stray mask pixels from becoming loose vertices in the .blend.
    cell = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
    cell = _break_diagonal_pinches(cell)
    if cell.sum() < 1:
        raise RuntimeError("silhouette too small to build a mesh")

    used = np.zeros_like(mask)
    used[:-1, :-1] |= cell
    used[:-1, 1:] |= cell
    used[1:, 1:] |= cell
    used[1:, :-1] |= cell

    back_scale = 0.55
    idx_front = -np.ones((grid, grid), dtype=np.int64)

    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    cell_of: list[tuple[int, int]] = []   # front vertex index -> (y, x)

    span = 2.0
    for y in range(grid):
        for x in range(grid):
            if not used[y, x]:
                continue
            px = (x / (grid - 1) - 0.5) * span
            # +Z is up in Blender; the image plane maps to X (right) and Z (up).
            pz = (0.5 - y / (grid - 1)) * span
            u = x / (grid - 1)
            v = 1.0 - y / (grid - 1)
            d = float(height[y, x]) * depth_scale

            # Front and back are written as a pair, so back == front + 1.
            idx_front[y, x] = len(verts)
            cell_of.append((y, x))
            verts.append((px, -d, pz))
            uvs.append((u, v))
            verts.append((px, d * back_scale, pz))
            uvs.append((u, v))

    def back_of(front_index: int) -> int:
        return front_index + 1

    def grid_of(front_index: int) -> tuple[int, int]:
        return cell_of[front_index // 2]

    # Front quads face -Y (toward the camera); back quads are the same loop
    # reversed, so they face +Y. Getting this order wrong turns the mesh
    # inside out: it still renders, but every normal points into the solid.
    faces: list[tuple[int, ...]] = []
    face_uvs: list[tuple[int, ...]] = []
    front_edges: set[tuple[int, int]] = set()

    def add_face(indices: tuple[int, ...], uv_indices: tuple[int, ...] | None = None) -> None:
        faces.append(indices)
        face_uvs.append(uv_indices if uv_indices is not None else indices)

    for y, x in zip(*np.nonzero(cell)):
        a = int(idx_front[y, x])
        b = int(idx_front[y + 1, x])
        c = int(idx_front[y + 1, x + 1])
        d = int(idx_front[y, x + 1])
        add_face((a, b, c, d))
        add_face((back_of(d), back_of(c), back_of(b), back_of(a)))
        quad = (a, b, c, d)
        for i in range(4):
            front_edges.add((quad[i], quad[(i + 1) % 4]))

    # Rim: exactly the front edges with no opposing twin, stitched to the back.
    # Traversing each in reverse is what makes every edge in the finished mesh
    # used once in each direction - the definition of a closed, oriented surface.
    #
    # The rim gets its own UVs, pulled a few pixels into the silhouette. Sampled
    # where they sit, they would land on the plain background the subject was
    # generated against, and every one of these meshes would wear a white edge.
    rim_uv_index: dict[int, int] = {}

    def rim_uv_for(front_index: int) -> int:
        cached = rim_uv_index.get(front_index)
        if cached is not None:
            return cached
        y, x = grid_of(front_index)
        for _ in range(RIM_UV_INSET):
            best = (dist[y, x], y, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid and 0 <= nx < grid and dist[ny, nx] > best[0]:
                        best = (dist[ny, nx], ny, nx)
            if (best[1], best[2]) == (y, x):
                break
            y, x = best[1], best[2]
        uvs.append((x / (grid - 1), 1.0 - y / (grid - 1)))
        rim_uv_index[front_index] = len(uvs) - 1
        return rim_uv_index[front_index]

    for u_idx, v_idx in front_edges:
        if (v_idx, u_idx) in front_edges:
            continue
        add_face(
            (v_idx, u_idx, back_of(u_idx), back_of(v_idx)),
            (rim_uv_for(v_idx), rim_uv_for(u_idx), rim_uv_for(u_idx), rim_uv_for(v_idx)),
        )

    if len(verts) < 8 or len(faces) < 4:
        raise RuntimeError("silhouette too small to build a mesh")

    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "texture.png"
    src.resize((1024, 1024), Image.LANCZOS).save(tex_path, format="PNG")

    mtl_path = out_dir / "mesh.mtl"
    mtl_path.write_text(
        "newmtl GenFXMaterial\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.100 0.100 0.100\n"
        "Ns 40.0\n"
        "d 1.0\n"
        "illum 2\n"
        f"map_Kd {tex_path.name}\n",
        encoding="utf-8",
    )

    obj_path = out_dir / "mesh.obj"
    with open(obj_path, "w", encoding="utf-8") as fh:
        fh.write("# GenFX generated solid\n")
        fh.write(f"mtllib {mtl_path.name}\n")
        fh.write("o GenFX_Subject\n")
        for vx, vy, vz in verts:
            fh.write(f"v {vx:.5f} {vy:.5f} {vz:.5f}\n")
        for u, v in uvs:
            fh.write(f"vt {u:.5f} {v:.5f}\n")
        fh.write("usemtl GenFXMaterial\n")
        fh.write("s 1\n")
        for face, face_uv in zip(faces, face_uvs):
            fh.write(
                "f " + " ".join(f"{v + 1}/{t + 1}" for v, t in zip(face, face_uv)) + "\n"
            )

    return obj_path, len(verts), len(faces)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_mesh(image_path: str, out_dir: str | Path) -> MeshResult:
    """
    Turn a reference image into a 3D mesh, walking the tier cascade.
    Never raises.
    """
    out_dir = Path(out_dir)
    attempts: list[str] = []
    last_error = ""
    started = time.time()

    for tier in config.MESH_PROVIDER_ORDER:
        try:
            if tier == "space":
                t0 = time.time()
                mesh_path, space_id, textured = generate_mesh_via_space(image_path, out_dir)
                attempts.append(f"space[{space_id}]: ok in {time.time() - t0:.1f}s")
                return MeshResult(
                    mesh_path=str(mesh_path),
                    status="ok",
                    method="space",
                    provider_used=space_id,
                    textured=textured,
                    attempts=attempts,
                    elapsed=time.time() - started,
                )

            if tier == "depth":
                img = Image.open(image_path).convert("RGB")
                depth = _estimate_depth_local(img)
                if depth is None:
                    raise RuntimeError("no depth backend (install transformers + torch)")
                t0 = time.time()
                obj, nv, nf = build_solid_from_image(image_path, out_dir, depth_map=depth)
                attempts.append(f"depth: ok in {time.time() - t0:.1f}s")
                return MeshResult(
                    mesh_path=str(obj),
                    status="ok",
                    method="depth",
                    provider_used="depth-anything",
                    textured=True,
                    attempts=attempts,
                    vertex_count=nv,
                    face_count=nf,
                    elapsed=time.time() - started,
                )

            if tier == "relief":
                t0 = time.time()
                obj, nv, nf = build_solid_from_image(image_path, out_dir)
                attempts.append(f"relief: ok in {time.time() - t0:.1f}s")
                return MeshResult(
                    mesh_path=str(obj),
                    status="ok" if not last_error else "fallback",
                    method="relief",
                    provider_used="local_inflation",
                    textured=True,
                    error_message=last_error or None,
                    attempts=attempts,
                    vertex_count=nv,
                    face_count=nf,
                    elapsed=time.time() - started,
                )

        except Exception as exc:
            last_error = f"{tier}: {type(exc).__name__} - {str(exc)[:200]}"
            attempts.append(last_error)
            logger.warning("Mesh tier failed - %s", last_error)

    return MeshResult(
        mesh_path=None,
        status="fallback",
        method="none",
        error_message=last_error or "all mesh tiers failed",
        attempts=attempts,
        elapsed=time.time() - started,
    )
