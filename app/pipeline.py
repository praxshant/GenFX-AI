"""
Pipeline orchestrator.

    prompt -> scene JSON -> reference image -> 3D mesh -> editable .blend

Every stage degrades instead of failing, so a run always ends with artifacts on
disk and a manifest explaining exactly which path each stage took.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app import config
from app.blend_builder import build_blend
from app.image_gen import generate_image
from app.llm_parser import parse_prompt
from app.mesh_gen import generate_mesh

logger = logging.getLogger(__name__)

STAGES = ("scene", "image", "mesh", "blend")


@dataclass
class RunResult:
    run_id: str
    prompt: str
    run_dir: str
    status: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, str | None] = field(default_factory=dict)
    providers: dict[str, str | None] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, list[str]] = field(default_factory=dict)

    scene_json: dict[str, Any] | None = None
    scene_json_path: str | None = None
    image_path: str | None = None
    mesh_path: str | None = None
    blend_path: str | None = None
    preview_path: str | None = None
    glb_path: str | None = None
    blend_stats: dict[str, Any] = field(default_factory=dict)

    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def ok(self) -> bool:
        """A run is a success when the artist actually gets a .blend."""
        return self.status.get("blend") == "ok"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        data["ok"] = self.ok
        return data


def _prune_old_runs(keep: int) -> None:
    try:
        runs = sorted(
            (p for p in config.RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in runs[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
    except FileNotFoundError:
        pass
    except Exception as exc:  # housekeeping must never break a run
        logger.warning("Run pruning failed: %s", exc)


def run_pipeline(
    user_prompt: str,
    on_stage: Callable[[str, str], None] | None = None,
    make_preview: bool | None = None,
) -> RunResult:
    """
    Execute the full pipeline.

    `on_stage(stage, state)` is called as each stage starts and settles, so a UI
    can show live progress. States are "running", "ok" and "fallback".
    """
    started = time.time()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        run_id=run_id,
        prompt=user_prompt,
        run_dir=str(run_dir),
        status={s: "pending" for s in STAGES},
        diagnostics={s: None for s in STAGES},
        providers={s: None for s in STAGES},
        timings={},
        attempts={},
        started_at=started,
    )

    def announce(stage: str, state: str) -> None:
        result.status[stage] = state
        if on_stage:
            try:
                on_stage(stage, state)
            except Exception:
                pass

    # ── Stage 1: prompt -> scene JSON ─────────────────────────────────────────
    announce("scene", "running")
    t0 = time.time()
    try:
        parsed = parse_prompt(user_prompt)
        result.scene_json = parsed.scene_json
        result.providers["scene"] = parsed.provider_used
        result.diagnostics["scene"] = parsed.error_message
        result.attempts["scene"] = parsed.attempts

        scene_path = run_dir / "scene.json"
        scene_path.write_text(json.dumps(parsed.scene_json, indent=2), encoding="utf-8")
        result.scene_json_path = str(scene_path)
        announce("scene", parsed.status)
    except Exception as exc:
        logger.exception("Stage 1 crashed")
        from app.llm_parser import normalise_scene

        result.scene_json = normalise_scene({}, user_prompt)
        result.diagnostics["scene"] = f"unhandled: {type(exc).__name__} - {exc}"
        announce("scene", "fallback")
    result.timings["scene"] = time.time() - t0

    # ── Stage 2: scene JSON -> reference image ────────────────────────────────
    announce("image", "running")
    t0 = time.time()
    try:
        image = generate_image(result.scene_json or {}, run_dir / "image.png", user_prompt=user_prompt)
        result.image_path = image.image_path
        result.providers["image"] = image.provider_used
        result.diagnostics["image"] = image.error_message
        result.attempts["image"] = image.attempts
        announce("image", image.status)
    except Exception as exc:
        logger.exception("Stage 2 crashed")
        result.image_path = str(config.FALLBACK_IMAGE_PATH)
        result.diagnostics["image"] = f"unhandled: {type(exc).__name__} - {exc}"
        announce("image", "fallback")
    result.timings["image"] = time.time() - t0

    # ── Stage 3: image -> mesh ────────────────────────────────────────────────
    announce("mesh", "running")
    t0 = time.time()
    try:
        mesh = generate_mesh(result.image_path, run_dir / "mesh")
        result.mesh_path = mesh.mesh_path
        result.providers["mesh"] = f"{mesh.method}:{mesh.provider_used}" if mesh.provider_used else mesh.method
        result.diagnostics["mesh"] = mesh.error_message
        result.attempts["mesh"] = mesh.attempts
        announce("mesh", mesh.status)
    except Exception as exc:
        logger.exception("Stage 3 crashed")
        result.diagnostics["mesh"] = f"unhandled: {type(exc).__name__} - {exc}"
        announce("mesh", "fallback")
    result.timings["mesh"] = time.time() - t0

    # ── Stage 4: mesh -> .blend ───────────────────────────────────────────────
    announce("blend", "running")
    t0 = time.time()
    try:
        if not result.mesh_path:
            raise RuntimeError("no mesh to build from")
        blend = build_blend(
            mesh_path=result.mesh_path,
            out_dir=run_dir,
            scene_json=result.scene_json,
            image_path=result.image_path,
            make_preview=make_preview,
        )
        result.blend_path = blend.blend_path
        result.preview_path = blend.preview_path
        result.glb_path = blend.glb_path
        result.blend_stats = blend.stats
        result.providers["blend"] = blend.runtime
        result.diagnostics["blend"] = blend.error_message
        announce("blend", blend.status)
    except Exception as exc:
        logger.exception("Stage 4 crashed")
        result.diagnostics["blend"] = f"unhandled: {type(exc).__name__} - {exc}"
        announce("blend", "fallback")
    result.timings["blend"] = time.time() - t0

    result.finished_at = time.time()

    manifest = run_dir / "manifest.json"
    try:
        manifest.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write manifest: %s", exc)

    logger.info(
        "Run %s finished in %.1fs | scene=%s image=%s mesh=%s blend=%s",
        run_id, result.duration, result.status["scene"], result.status["image"],
        result.status["mesh"], result.status["blend"],
    )

    _prune_old_runs(config.MAX_RUNS_KEPT)
    return result


def load_run(run_id: str) -> RunResult | None:
    """Rehydrate a previous run from its manifest."""
    manifest = config.RUNS_DIR / run_id / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data.pop("duration", None)
        data.pop("ok", None)
        return RunResult(**data)
    except Exception as exc:
        logger.warning("Could not load run %s: %s", run_id, exc)
        return None


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Recent runs, newest first, for the gallery."""
    if not config.RUNS_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    dirs = sorted(
        (p for p in config.RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in dirs[:limit]:
        manifest = path / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            out.append({
                "run_id": data.get("run_id", path.name),
                "prompt": data.get("prompt", ""),
                "preview_path": data.get("preview_path"),
                "blend_path": data.get("blend_path"),
                "glb_path": data.get("glb_path"),
                "image_path": data.get("image_path"),
                "status": data.get("status", {}),
                "duration": data.get("duration", 0.0),
            })
        except Exception:
            continue
    return out
