"""
GenFX test suite.

Everything here runs offline: no API keys, no network, no Blender. Tests that
need a real Blender runtime are skipped rather than failed.

    pytest -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.blend_builder import build_blend, describe_runtime, resolve_blender_path  # noqa: E402
from app.image_gen import (  # noqa: E402
    ImageResult,
    build_image_prompt,
    generate_image,
    prepare_for_reconstruction,
)
from app.llm_parser import (  # noqa: E402
    ParserResult,
    ProviderUnavailable,
    ValidationError,
    _extract_json,
    build_heuristic_image_prompt,
    normalise_scene,
    parse_prompt,
    validate_schema,
)
from app.mesh_gen import (  # noqa: E402
    MeshResult,
    _harvest_mesh_file,
    build_solid_from_image,
    distance_transform,
    foreground_mask,
    generate_mesh,
)
from app.pipeline import list_runs, run_pipeline  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _load_build_blend_module():
    spec = importlib.util.spec_from_file_location(
        "genfx_build_blend", PROJECT_ROOT / "blender" / "build_blend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def build_blend_module():
    """The Blender script is importable without Blender - that is the point."""
    return _load_build_blend_module()


@pytest.fixture
def subject_image(tmp_path: Path) -> Path:
    """A centred blob on a plain background, like stage 2 is asked to produce."""
    img = Image.new("RGB", (256, 256), (238, 238, 240))
    pixels = img.load()
    cx, cy = 128, 130
    for y in range(256):
        for x in range(256):
            if ((x - cx) / 62) ** 2 + ((y - cy) / 78) ** 2 <= 1.0:
                shade = 90 + int(70 * (1 - abs(x - cx) / 62))
                pixels[x, y] = (shade, int(shade * 0.55), 40)
    path = tmp_path / "subject.png"
    img.save(path)
    return path


@pytest.fixture
def scene() -> dict:
    return normalise_scene({}, "a carved wooden owl figurine")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 - prompt parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestJsonExtraction:
    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_multiline_inside_code_fence(self):
        """The historical failure: fenced, multi-line JSON must survive."""
        raw = '```json\n{\n  "scene_id": "sc_1",\n  "n": 2\n}\n```'
        assert _extract_json(raw) == {"scene_id": "sc_1", "n": 2}

    def test_fence_without_language_tag(self):
        assert _extract_json('```\n{\n  "a": [1, 2]\n}\n```') == {"a": [1, 2]}

    def test_prose_before_and_after(self):
        raw = 'Sure! Here is the scene:\n{\n "a": 1\n}\nLet me know if you need changes.'
        assert _extract_json(raw) == {"a": 1}

    def test_trailing_comma_is_repaired(self):
        assert _extract_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        raw = '{"image_prompt": "a sign that says {hello}", "n": 1}'
        assert _extract_json(raw)["n"] == 1

    def test_prefers_the_largest_object(self):
        raw = '{"partial": 1}\n{"scene_id": "sc", "environment": {"type": "studio"}}'
        assert "scene_id" in _extract_json(raw)

    def test_empty_response_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("   ")

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("I cannot help with that request.")


class TestSchema:
    def test_normalised_scene_validates(self, scene):
        assert validate_schema(scene) is True

    def test_missing_key_raises(self):
        with pytest.raises(ValidationError, match="Missing required key"):
            validate_schema({"scene_id": "x"})

    def test_wrong_nested_type_raises(self, scene):
        scene["lighting"] = "bright"
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_schema(scene)

    def test_empty_image_prompt_raises(self, scene):
        scene["image_prompt"] = "  "
        with pytest.raises(ValidationError, match="image_prompt"):
            validate_schema(scene)


class TestNormalisation:
    def test_fills_every_required_key(self):
        out = normalise_scene({}, "a brass lamp")
        for key in ("subject", "environment", "lighting", "camera", "materials",
                    "render_settings", "image_prompt", "asset_refs"):
            assert key in out

    def test_clamps_out_of_range_values(self):
        out = normalise_scene(
            {"lighting": {"intensity": 900}, "materials": {"roughness": -4, "metallic": 12},
             "camera": {"focal_length": 9000}},
            "x",
        )
        assert out["lighting"]["intensity"] <= 5.0
        assert 0.0 <= out["materials"]["roughness"] <= 1.0
        assert 0.0 <= out["materials"]["metallic"] <= 1.0
        assert out["camera"]["focal_length"] <= 300

    def test_expands_short_hex_and_rejects_junk(self):
        out = normalise_scene({"materials": {"base_color": "#f0a"},
                               "lighting": {"key_color": "not a colour"}}, "x")
        assert out["materials"]["base_color"] == "#ff00aa"
        assert out["lighting"]["key_color"] == "#ffffff"

    def test_asset_refs_always_empty(self):
        assert normalise_scene({"asset_refs": ["a", "b"]}, "x")["asset_refs"] == []

    def test_reconstruction_terms_appended(self):
        out = normalise_scene({"image_prompt": "a red mug"}, "a red mug")
        assert "centered" in out["image_prompt"]

    def test_survives_legacy_scene_shape(self):
        legacy = {
            "scene_id": "sc_001",
            "environment": {"type": "desert", "time_of_day": "golden_hour"},
            "lighting": {"preset": "cinematic_sunset", "intensity": 0.85},
            "camera": {"shot_type": "wide", "angle": "low"},
            "effects": [{"type": "dust"}],
            "render_settings": {"resolution": "1920x1080", "samples": 64},
            "asset_refs": [],
        }
        assert validate_schema(normalise_scene(legacy, "a desert")) is True


class TestProviderCascade:
    def test_no_providers_falls_back_locally(self):
        with patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "OLLAMA_ENABLED", False):
            result = parse_prompt("a snowy mountain cabin")

        assert isinstance(result, ParserResult)
        assert result.status == "fallback"
        assert result.provider_used == "local_heuristic"
        assert validate_schema(result.scene_json) is True
        # The prompt still reaches stage 2 even with no LLM at all.
        assert "snowy mountain cabin" in result.scene_json["image_prompt"]

    def test_falls_through_to_the_next_provider(self):
        good = json.dumps({
            "scene_id": "sc_2", "subject": {"name": "lamp"},
            "environment": {"type": "studio"}, "lighting": {"intensity": 1},
            "camera": {}, "materials": {}, "effects": [],
            "render_settings": {}, "image_prompt": "a lamp", "asset_refs": [],
        })
        with patch.object(config, "LLM_PROVIDER_ORDER", ["openrouter", "openai"]), \
             patch("app.llm_parser._parse_with_openrouter", side_effect=RuntimeError("503")), \
             patch("app.llm_parser._parse_with_openai", return_value=good):
            result = parse_prompt("a lamp")

        assert result.status == "ok"
        assert result.provider_used == "openai"

    def test_unavailable_provider_is_not_retried(self):
        calls = []

        def unavailable(_prompt):
            calls.append(1)
            raise ProviderUnavailable("no key")

        with patch.object(config, "LLM_PROVIDER_ORDER", ["openrouter"]), \
             patch.object(config, "LLM_RETRY_COUNT", 3), \
             patch("app.llm_parser._parse_with_openrouter", unavailable):
            parse_prompt("a lamp")

        assert len(calls) == 1

    def test_bad_json_then_fallback(self):
        with patch.object(config, "LLM_PROVIDER_ORDER", ["openrouter"]), \
             patch.object(config, "LLM_RETRY_COUNT", 0), \
             patch("app.llm_parser._parse_with_openrouter", return_value="I'm sorry Dave"):
            result = parse_prompt("a lamp")

        assert result.status == "fallback"
        assert "invalid JSON" in (result.error_message or "")

    def test_ollama_reports_unavailable_when_daemon_is_absent(self):
        from app.llm_parser import _parse_with_ollama

        with patch("requests.get", side_effect=OSError("connection refused")):
            with pytest.raises(ProviderUnavailable):
                _parse_with_ollama("a lamp")

    def test_ollama_reports_unavailable_when_model_missing(self):
        from app.llm_parser import _parse_with_ollama

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
        with patch("requests.get", return_value=response), \
             patch.object(config, "OLLAMA_MODEL", "llama3.2"):
            with pytest.raises(ProviderUnavailable, match="not pulled"):
                _parse_with_ollama("a lamp")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 - image generation
# ══════════════════════════════════════════════════════════════════════════════

class TestImagePrompt:
    def test_authored_prompt_wins(self, scene):
        scene["image_prompt"] = "a hand-written prompt"
        assert build_image_prompt(scene, user_prompt="ignored") == "a hand-written prompt"

    def test_user_prompt_used_when_scene_has_none(self):
        prompt = build_image_prompt({}, user_prompt="a snowy mountain valley")
        assert "snowy mountain valley" in prompt

    def test_built_from_scene_fields_as_last_resort(self):
        prompt = build_image_prompt({"subject": {"name": "a brass compass"}})
        assert "brass compass" in prompt

    def test_heuristic_prompt_includes_reconstruction_terms(self):
        assert "plain seamless background" in build_heuristic_image_prompt("a mug")


class TestImagePostProcessing:
    def test_wide_image_is_squared_without_cropping(self):
        img = Image.new("RGB", (1024, 512), (200, 30, 30))
        out = prepare_for_reconstruction(img, size=512)
        assert out.size == (512, 512)

    def test_square_image_passes_through(self):
        out = prepare_for_reconstruction(Image.new("RGB", (300, 300)), size=256)
        assert out.size == (256, 256)


class TestImageGeneration:
    def test_first_provider_success(self, tmp_path):
        buf = Image.new("RGB", (64, 64), (10, 200, 90))
        import io

        raw = io.BytesIO()
        buf.save(raw, format="PNG")

        with patch.object(config, "IMAGE_PROVIDER_ORDER", ["pollinations"]), \
             patch("app.image_gen._generate_pollinations", return_value=raw.getvalue()):
            result = generate_image({}, tmp_path / "img.png", user_prompt="a mug")

        assert isinstance(result, ImageResult)
        assert result.status == "ok"
        assert result.provider_used == "pollinations"
        assert Path(result.image_path).exists()

    def test_all_providers_fail_returns_fallback_asset(self, tmp_path):
        with patch.object(config, "IMAGE_PROVIDER_ORDER", ["pollinations", "huggingface"]), \
             patch.object(config, "IMAGE_RETRY_COUNT", 0), \
             patch("app.image_gen._generate_pollinations", side_effect=RuntimeError("503")), \
             patch("app.image_gen._generate_huggingface", side_effect=RuntimeError("no key")):
            result = generate_image({}, tmp_path / "img.png", user_prompt="a mug")

        assert result.status == "fallback"
        assert result.image_path.endswith("fallback_image.png")
        assert len(result.attempts) == 2

    def test_second_provider_rescues_the_first(self, tmp_path):
        import io

        raw = io.BytesIO()
        Image.new("RGB", (32, 32)).save(raw, format="PNG")

        with patch.object(config, "IMAGE_PROVIDER_ORDER", ["pollinations", "huggingface"]), \
             patch.object(config, "IMAGE_RETRY_COUNT", 0), \
             patch("app.image_gen._generate_pollinations", side_effect=RuntimeError("429")), \
             patch("app.image_gen._generate_huggingface", return_value=raw.getvalue()):
            result = generate_image({}, tmp_path / "img.png", user_prompt="a mug")

        assert result.status == "ok"
        assert result.provider_used == "huggingface"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 - mesh generation
# ══════════════════════════════════════════════════════════════════════════════

class TestSegmentation:
    def test_finds_the_subject_against_a_plain_background(self, subject_image):
        arr = np.asarray(Image.open(subject_image).convert("RGB"))
        mask = foreground_mask(arr)
        assert 0.05 < mask.mean() < 0.6
        assert mask[130, 128]        # centre of the blob
        assert not mask[4, 4]        # corner is background

    def test_uniform_image_degrades_to_full_frame(self):
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert foreground_mask(arr).mean() == 1.0

    def test_distance_transform_peaks_in_the_middle(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        dist = distance_transform(mask)
        assert dist[20, 20] > dist[11, 11]
        assert dist[0, 0] == 0.0


class TestSolidConstruction:
    def test_builds_a_textured_obj(self, subject_image, tmp_path):
        obj, verts, faces = build_solid_from_image(
            str(subject_image), tmp_path / "mesh", grid=64
        )
        assert obj.exists() and obj.suffix == ".obj"
        assert (tmp_path / "mesh" / "mesh.mtl").exists()
        assert (tmp_path / "mesh" / "texture.png").exists()
        assert verts > 100 and faces > 50

        text = obj.read_text()
        assert text.count("\nv ") == verts
        assert "usemtl GenFXMaterial" in text
        assert "vt " in text  # UVs, so the texture actually lands

    def test_has_front_and_back_so_it_is_a_solid(self, subject_image, tmp_path):
        obj, verts, _ = build_solid_from_image(str(subject_image), tmp_path / "m", grid=48)
        ys = [
            float(line.split()[2])
            for line in obj.read_text().splitlines()
            if line.startswith("v ")
        ]
        assert min(ys) < 0 < max(ys), "mesh should have depth on both sides"


class TestMeshHarvesting:
    def test_picks_the_glb_over_the_obj(self, tmp_path):
        glb = tmp_path / "src.glb"
        glb.write_bytes(b"glTF" + b"\x00" * 600)
        obj = tmp_path / "src.obj"
        obj.write_text("v 0 0 0\n" * 200)

        found = _harvest_mesh_file([str(obj), str(glb)], "https://x.hf.space", tmp_path / "out")
        assert found is not None
        path, _ = found
        assert path.suffix == ".glb"

    def test_returns_none_when_there_is_no_mesh(self, tmp_path):
        assert _harvest_mesh_file(["<div>html</div>", 42], "https://x", tmp_path) is None

    def test_white_mesh_is_marked_untextured(self, tmp_path):
        src = tmp_path / "white_mesh.glb"
        src.write_bytes(b"glTF" + b"\x00" * 600)
        path, textured = _harvest_mesh_file([str(src)], "https://x", tmp_path / "o")
        assert path.exists()
        assert textured is False


class TestMeshCascade:
    def test_falls_back_to_local_construction(self, subject_image, tmp_path):
        with patch.object(config, "MESH_PROVIDER_ORDER", ["space", "depth", "relief"]), \
             patch("app.mesh_gen.generate_mesh_via_space", side_effect=RuntimeError("no quota")), \
             patch("app.mesh_gen._estimate_depth_local", return_value=None), \
             patch("app.mesh_gen._estimate_depth_hf", return_value=None), \
             patch.object(config, "RELIEF_GRID", 60):
            result = generate_mesh(str(subject_image), tmp_path / "mesh")

        assert isinstance(result, MeshResult)
        assert result.method == "relief"
        assert result.textured is True
        assert Path(result.mesh_path).exists()
        assert any("space" in a for a in result.attempts)

    def test_space_success_short_circuits(self, subject_image, tmp_path):
        fake = tmp_path / "fake.glb"
        fake.write_bytes(b"glTF" + b"\x00" * 600)

        with patch.object(config, "MESH_PROVIDER_ORDER", ["space", "relief"]), \
             patch("app.mesh_gen.generate_mesh_via_space",
                   return_value=(fake, "frogleo/Image-to-3D", False)):
            result = generate_mesh(str(subject_image), tmp_path / "mesh")

        assert result.method == "space"
        assert result.provider_used == "frogleo/Image-to-3D"

    def test_every_tier_failing_is_reported_not_raised(self, tmp_path):
        missing = tmp_path / "nope.png"
        with patch.object(config, "MESH_PROVIDER_ORDER", ["space", "relief"]), \
             patch("app.mesh_gen.generate_mesh_via_space", side_effect=RuntimeError("down")):
            result = generate_mesh(str(missing), tmp_path / "mesh")

        assert result.status == "fallback"
        assert result.mesh_path is None
        assert result.error_message


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 - .blend building
# ══════════════════════════════════════════════════════════════════════════════

class TestBlenderScriptPure:
    """The builder script must be importable and testable without Blender."""

    def test_parses_args_after_the_double_dash(self, build_blend_module):
        args = build_blend_module.parse_args(
            ["--background", "--", "--mesh", "a.glb", "--output", "b.blend"]
        )
        assert args.mesh == "a.glb" and args.output == "b.blend"

    def test_parses_args_without_the_double_dash(self, build_blend_module):
        args = build_blend_module.parse_args(["--mesh", "a.obj", "--output", "b.blend"])
        assert args.mesh == "a.obj"

    def test_hex_to_linear_rgba(self, build_blend_module):
        r, g, b, a = build_blend_module.hex_to_rgba("#ffffff")
        assert a == 1.0
        assert all(abs(c - 1.0) < 1e-6 for c in (r, g, b))
        assert build_blend_module.hex_to_rgba("#000000")[:3] == (0.0, 0.0, 0.0)

    def test_short_hex_and_garbage_are_handled(self, build_blend_module):
        assert len(build_blend_module.hex_to_rgba("#f00")) == 4
        assert len(build_blend_module.hex_to_rgba("purple-ish")) == 4

    def test_scene_json_loader_tolerates_junk(self, build_blend_module, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert build_blend_module.load_scene_json(str(bad)) == {}
        assert build_blend_module.load_scene_json(None) == {}

    def test_known_lighting_presets_are_complete(self, build_blend_module):
        for preset in build_blend_module.LIGHT_PRESETS.values():
            assert {"key", "fill", "rim", "world"} <= set(preset)


class TestBlendBuilder:
    def test_missing_mesh_is_reported(self, tmp_path):
        result = build_blend(tmp_path / "nope.glb", tmp_path / "out")
        assert result.status == "fallback"
        assert "not found" in result.error_message

    def test_resolve_blender_path_handles_absent_binary(self):
        assert resolve_blender_path("definitely-not-a-real-binary-xyz") is None

    def test_describe_runtime_shape(self):
        runtime = describe_runtime()
        assert {"blender_binary", "bpy_module", "worker"} <= set(runtime)

    def test_no_runtime_degrades_gracefully(self, tmp_path):
        mesh = tmp_path / "m.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        with patch("app.blend_builder.resolve_blender_path", return_value=None), \
             patch("app.blend_builder._run", side_effect=RuntimeError("no bpy")), \
             patch.object(config, "BLEND_WORKER_URL", ""):
            result = build_blend(mesh, tmp_path / "out")

        assert result.status == "fallback"
        assert result.runtime == "none"
        assert result.blend_path is None


def _has_blend_runtime() -> bool:
    if resolve_blender_path():
        return True
    return importlib.util.find_spec("bpy") is not None


@pytest.mark.skipif(not _has_blend_runtime(), reason="no Blender runtime available")
class TestBlendBuilderIntegration:
    def test_builds_an_openable_blend_from_an_obj(self, subject_image, tmp_path):
        obj, _, _ = build_solid_from_image(str(subject_image), tmp_path / "mesh", grid=48)
        result = build_blend(
            obj, tmp_path / "out",
            scene_json=normalise_scene({}, "test subject"),
            image_path=str(subject_image),
            make_preview=False,
        )

        assert result.status == "ok", result.error_message
        blend = Path(result.blend_path)
        assert blend.exists() and blend.stat().st_size > 1024
        # Raw .blend starts with "BLENDER"; compressed ones are Zstandard
        # (Blender 3.0+) or gzip (older).
        head = blend.read_bytes()[:8]
        assert (
            head.startswith(b"BLENDER")
            or head[:4] == b"\x28\xb5\x2f\xfd"   # zstd
            or head[:2] == b"\x1f\x8b"           # gzip
        ), f"unexpected .blend header: {head!r}"
        assert result.stats.get("polygons", 0) > 0

    def test_the_saved_blend_reopens_with_the_expected_scene(self, subject_image, tmp_path):
        """The deliverable is a file an artist opens - so open it and check."""
        obj, _, _ = build_solid_from_image(str(subject_image), tmp_path / "mesh", grid=48)
        result = build_blend(
            obj, tmp_path / "out",
            scene_json=normalise_scene({"materials": {"base_color": "#c0392b"}}, "test"),
            image_path=str(subject_image),
            make_preview=False,
        )
        assert result.status == "ok", result.error_message

        probe = tmp_path / "probe.py"
        probe.write_text(
            "import bpy, json, sys\n"
            f"bpy.ops.wm.open_mainfile(filepath={str(result.blend_path)!r})\n"
            "meshes = [o for o in bpy.data.objects if o.type == 'MESH']\n"
            "subject = next((o for o in meshes if o.name == 'GenFX_Subject'), None)\n"
            "print('PROBE ' + json.dumps({\n"
            "    'subject': subject is not None,\n"
            "    'verts': len(subject.data.vertices) if subject else 0,\n"
            "    'materials': len(subject.data.materials) if subject else 0,\n"
            "    'lights': len([o for o in bpy.data.objects if o.type == 'LIGHT']),\n"
            "    'camera': bpy.context.scene.camera is not None,\n"
            "    'packed_images': len([i for i in bpy.data.images if i.packed_file]),\n"
            "}))\n"
        )

        import subprocess

        blender = resolve_blender_path()
        cmd = (
            [blender, "--background", "--factory-startup", "--python", str(probe)]
            if blender else [sys.executable, str(probe)]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE ")), None
        )
        assert line, f"probe produced no result:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"

        state = json.loads(line[len("PROBE "):])
        assert state["subject"], "GenFX_Subject missing from the reopened file"
        assert state["verts"] > 100
        assert state["materials"] >= 1
        assert state["lights"] >= 3, "three-point rig should survive the round trip"
        assert state["camera"], "scene camera should be set"
        assert state["packed_images"] >= 1, "textures should be packed into the .blend"


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

class TestPipeline:
    def test_full_run_with_every_stage_degraded(self, tmp_path):
        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", ["pollinations"]), \
             patch.object(config, "IMAGE_RETRY_COUNT", 0), \
             patch("app.image_gen._generate_pollinations", side_effect=RuntimeError("offline")), \
             patch.object(config, "MESH_PROVIDER_ORDER", ["space"]), \
             patch("app.mesh_gen.generate_mesh_via_space", side_effect=RuntimeError("offline")):
            result = run_pipeline("a brass telescope")

        assert result.status["scene"] == "fallback"
        assert result.status["image"] == "fallback"
        assert result.status["mesh"] == "fallback"
        assert result.status["blend"] == "fallback"
        assert result.ok is False
        assert Path(result.run_dir, "manifest.json").exists()
        assert Path(result.run_dir, "scene.json").exists()
        for stage in ("scene", "image", "mesh", "blend"):
            assert result.diagnostics[stage], f"{stage} should explain itself"

    def test_stage_callback_reports_progress(self, tmp_path):
        seen: list[tuple[str, str]] = []

        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []), \
             patch.object(config, "MESH_PROVIDER_ORDER", []):
            run_pipeline("a mug", on_stage=lambda s, st: seen.append((s, st)))

        stages = [s for s, _ in seen]
        assert stages.count("scene") >= 2  # running, then settled
        assert {"scene", "image", "mesh", "blend"} <= set(stages)

    def test_a_crashing_callback_cannot_break_a_run(self, tmp_path):
        def boom(_stage, _state):
            raise ValueError("UI exploded")

        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []), \
             patch.object(config, "MESH_PROVIDER_ORDER", []):
            result = run_pipeline("a mug", on_stage=boom)

        assert result.run_id.startswith("run_")

    def test_runs_are_isolated_from_each_other(self, tmp_path):
        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []), \
             patch.object(config, "MESH_PROVIDER_ORDER", []):
            a = run_pipeline("a mug")
            b = run_pipeline("a lamp")
            listed = list_runs(limit=5)

        assert a.run_id != b.run_id
        assert a.run_dir != b.run_dir
        assert {a.run_id, b.run_id} <= {entry["run_id"] for entry in listed}

    def test_old_runs_are_pruned(self, tmp_path):
        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "MAX_RUNS_KEPT", 2), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []), \
             patch.object(config, "MESH_PROVIDER_ORDER", []):
            for i in range(4):
                run_pipeline(f"object {i}")
            remaining = list((tmp_path / "runs").iterdir())

        assert len(remaining) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_probes_never_raise_without_network(self):
        from app.health import check_runtime_health

        with patch("app.health.requests.get", side_effect=OSError("no network")), \
             patch("app.health.requests.head", side_effect=OSError("no network")):
            health = check_runtime_health()

        assert set(health) == {"llm", "image", "mesh", "blend", "assets"}
        for probe in health.values():
            assert isinstance(probe["ok"], bool)
            assert "error" in probe and "detail" in probe

    def test_llm_probe_reports_no_provider(self):
        from app.health import check_llm

        with patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "OLLAMA_ENABLED", False):
            probe = check_llm()

        assert probe["ok"] is False
        assert "heuristic" in probe["detail"]

    def test_assets_probe_passes_on_a_fresh_checkout(self):
        from app.health import check_assets

        probe = check_assets()
        assert probe["ok"] is True, probe["error"]


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_summary_leaks_no_secrets(self):
        blob = json.dumps(config.summary())
        for secret in (config.OPENROUTER_API_KEY, config.OPENAI_API_KEY, config.HUGGINGFACE_API_KEY):
            if secret:
                assert secret not in blob

    def test_cascade_orders_are_non_empty(self):
        assert config.LLM_PROVIDER_ORDER
        assert config.IMAGE_PROVIDER_ORDER
        assert config.MESH_PROVIDER_ORDER

    def test_blend_builder_script_exists(self):
        assert config.BLEND_BUILDER_SCRIPT.exists()
