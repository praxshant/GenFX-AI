"""
GenFX test suite.

Everything here runs offline: no API keys, no network, no Blender. Tests that
need a real Blender runtime are skipped rather than failed.

    pytest -q
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from collections import Counter
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
    _break_diagonal_pinches,
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

    @staticmethod
    def _http(status: int, body: dict | None = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.text = json.dumps(body or {})
        resp.json.return_value = body or {}
        if status >= 400:
            import requests

            resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
        else:
            resp.raise_for_status.return_value = None
        return resp

    def _keys_everywhere(self):
        return (
            patch.object(config, "LLM_PROVIDER_ORDER", ["openrouter", "openai", "huggingface", "ollama"]),
            patch.object(config, "OPENROUTER_API_KEY", "sk-or-bad"),
            patch.object(config, "OPENAI_API_KEY", "sk-bad"),
            patch.object(config, "HUGGINGFACE_API_KEY", "hf_bad"),
            patch.object(config, "OLLAMA_ENABLED", True),
            patch.object(config, "OLLAMA_MODEL", "llama3.2"),
            patch.object(config, "LLM_RETRY_COUNT", 2),
        )

    def test_every_key_rejected_lands_on_ollama(self):
        """
        The case this cascade exists for: keys are set but none works. Each
        keyed provider gets one call - a 401 will not change on retry - and
        the local Ollama daemon answers.
        """
        scene = normalise_scene({}, "a lamp")
        posted: list[str] = []

        def post(url, **_kwargs):
            posted.append(url)
            if "11434" in url:
                return self._http(200, {"message": {"content": json.dumps(scene)}})
            return self._http(401, {"error": "invalid key"})

        tags = self._http(200, {"models": [{"name": "llama3.2:latest"}]})
        with contextlib.ExitStack() as stack:
            for p in self._keys_everywhere():
                stack.enter_context(p)
            stack.enter_context(patch("requests.post", side_effect=post))
            stack.enter_context(patch("requests.get", return_value=tags))
            result = parse_prompt("a lamp")

        assert result.status == "ok"
        assert result.provider_used == "ollama"
        keyed = [u for u in posted if "11434" not in u]
        assert len(keyed) == 3, f"rejected keys were retried: {keyed}"
        assert any("key rejected" in a for a in result.attempts)

    def test_every_key_rejected_and_no_ollama_uses_the_local_parser(self):
        """No working key and no Ollama on the machine: still a usable scene."""
        with contextlib.ExitStack() as stack:
            for p in self._keys_everywhere():
                stack.enter_context(p)
            stack.enter_context(patch("requests.post", return_value=self._http(401)))
            stack.enter_context(patch("requests.get", side_effect=OSError("connection refused")))
            result = parse_prompt("a lamp")

        assert result.status == "fallback"
        assert result.provider_used == "local_heuristic"
        assert validate_schema(result.scene_json) is True
        assert [a.split(":")[0] for a in result.attempts] == [
            "openrouter", "openai", "huggingface", "ollama",
        ]
        assert "Ollama not reachable" in result.error_message

    def test_a_server_error_is_still_retried(self):
        """Unlike a rejected key, a 503 may clear up on the next attempt."""
        calls = []

        def post(url, **_kwargs):
            calls.append(url)
            return self._http(503)

        with patch.object(config, "LLM_PROVIDER_ORDER", ["openai"]), \
             patch.object(config, "OPENAI_API_KEY", "sk-x"), \
             patch.object(config, "LLM_RETRY_COUNT", 2), \
             patch("requests.post", side_effect=post):
            parse_prompt("a lamp")

        assert len(calls) == 3

    @pytest.mark.parametrize("model, installed, expected", [
        ("llama3.2", {"llama3.2:latest"}, True),
        ("llama3.2", {"llama3.2:3b"}, True),
        ("llama3.2:1b", {"llama3.2:3b"}, False),
        ("llama3.2:1b", {"llama3.2:1b"}, True),
        ("llama3.2", set(), False),
        ("llama3.2", {"llama3.1:latest"}, False),
    ])
    def test_ollama_model_matching(self, model, installed, expected):
        from app.llm_parser import _ollama_has_model

        assert _ollama_has_model(model, installed) is expected

    def test_ollama_with_nothing_pulled_is_unavailable(self):
        from app.llm_parser import _parse_with_ollama

        with patch("requests.get", return_value=self._http(200, {"models": []})), \
             patch("requests.post") as post:
            with pytest.raises(ProviderUnavailable, match="not pulled"):
                _parse_with_ollama("a lamp")
        post.assert_not_called()

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

    def test_unconfigured_provider_is_not_retried(self, tmp_path):
        """No key will not appear between attempts - retrying only adds sleeps."""
        from app.image_gen import ProviderUnavailable as ImageProviderUnavailable

        calls = []

        def unavailable(*_args):
            calls.append(1)
            raise ImageProviderUnavailable("no key")

        with patch.object(config, "IMAGE_PROVIDER_ORDER", ["huggingface"]), \
             patch.object(config, "IMAGE_RETRY_COUNT", 3), \
             patch("app.image_gen._generate_huggingface", unavailable), \
             patch("app.image_gen.time.sleep") as sleep:
            result = generate_image({}, tmp_path / "img.png", user_prompt="a mug")

        assert len(calls) == 1
        sleep.assert_not_called()
        assert result.status == "fallback"


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

    def test_survives_a_gradient_backdrop(self):
        """
        Text-to-image models produce studio gradients, not flat colour. Measured
        against a single median colour, half the backdrop reads as subject and
        the mesh comes out a slab.
        """
        h = w = 160
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):  # dark above, warm below
            shade = 40 + int(150 * y / h)
            arr[y, :] = (shade, int(shade * 0.95), int(shade * 0.85))
        for y in range(h):
            for x in range(w):
                if ((x - 80) / 26) ** 2 + ((y - 84) / 34) ** 2 <= 1.0:
                    arr[y, x] = (20, 90, 200)

        mask = foreground_mask(arr)
        assert 0.02 < mask.mean() < 0.20, f"backdrop leaked in: {mask.mean():.1%}"
        assert mask[84, 80], "subject centre was not selected"
        assert not mask[6, 6] and not mask[h - 6, w - 6], "backdrop corners selected"

    def test_a_centred_subject_beats_a_larger_backdrop_patch(self):
        """Picking the biggest blob hands back the wall, not the object."""
        arr = np.full((120, 120, 3), 235, dtype=np.uint8)
        arr[:40, :] = (30, 30, 30)                 # a big dark band along the top
        arr[50:80, 45:75] = (200, 40, 40)          # the smaller, centred subject

        mask = foreground_mask(arr)
        assert mask[64, 60], "centred subject not selected"
        assert not mask[10, 60], "the larger backdrop band won instead"

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


def _read_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append(tuple(float(t) for t in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return verts, faces


@pytest.fixture(scope="module")
def obj(tmp_path_factory) -> Path:
    """One solid, built once, inspected from several angles below."""
    tmp = tmp_path_factory.mktemp("solid")
    img = Image.new("RGB", (192, 192), (238, 238, 240))
    pixels = img.load()
    for y in range(192):
        for x in range(192):
            if ((x - 96) / 48) ** 2 + ((y - 98) / 62) ** 2 <= 1.0:
                pixels[x, y] = (150, 80, 40)
    src = tmp / "subject.png"
    img.save(src)
    path, _, _ = build_solid_from_image(str(src), tmp / "mesh", grid=56)
    return path


class TestSolidIsAProperSolid:
    """
    The three properties that decide whether the .blend is usable rather than
    merely openable. All three were broken once; none is cheap to spot by eye.
    """

    def test_watertight(self, obj):
        """Every edge borders exactly two faces - no holes for Blender to find."""
        _, faces = _read_obj(obj)
        edges = Counter()
        for face in faces:
            for i in range(len(face)):
                a, b = face[i], face[(i + 1) % len(face)]
                edges[(min(a, b), max(a, b))] += 1
        assert [e for e, n in edges.items() if n == 1] == [], "open boundary edges"
        assert [e for e, n in edges.items() if n > 2] == [], "non-manifold edges"

    def test_consistently_wound(self, obj):
        """Each directed edge used once, which is what makes the surface oriented."""
        _, faces = _read_obj(obj)
        directed = Counter()
        for face in faces:
            for i in range(len(face)):
                directed[(face[i], face[(i + 1) % len(face)])] += 1
        assert max(directed.values()) == 1

    def test_normals_point_outward(self, obj):
        """
        Positive signed volume. Negative means the solid is inside out: it still
        renders, but the light rig lands on the back of every face.
        """
        verts, faces = _read_obj(obj)
        volume = 0.0
        for face in faces:
            for k in range(1, len(face) - 1):
                a, b, c = verts[face[0]], verts[face[k]], verts[face[k + 1]]
                volume += (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                ) / 6.0
        assert volume > 0, "mesh is inside out"

    def test_no_loose_vertices(self, obj):
        verts, faces = _read_obj(obj)
        assert {i for face in faces for i in face} == set(range(len(verts)))

    def test_rim_uvs_are_pulled_inside_the_silhouette(self, obj):
        """
        The rim samples the texture a few pixels in. Left on the silhouette it
        would sample the plain background and every mesh would wear a white edge.
        """
        text = obj.read_text()
        uv_count = text.count("\nvt ")
        vert_count = text.count("\nv ")
        assert uv_count > vert_count, "no extra rim UVs were emitted"

    def test_diagonal_pinches_are_removed(self):
        """Two cells meeting only at a corner would make that point non-manifold."""
        cell = np.zeros((6, 6), dtype=bool)
        cell[1, 1] = cell[2, 2] = True
        cleaned = _break_diagonal_pinches(cell)
        assert cleaned.sum() < 2


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
             patch.object(config, "RELIEF_GRID", 60):
            result = generate_mesh(str(subject_image), tmp_path / "mesh")

        assert isinstance(result, MeshResult)
        assert result.method == "relief"
        assert result.textured is True
        assert Path(result.mesh_path).exists()
        assert any("space" in a for a in result.attempts)

    def test_depth_tier_accepts_a_real_depth_array(self, subject_image, tmp_path):
        """
        `local() or hf()` asked numpy for the truth value of a whole depth map,
        which raises - so this tier could only ever fail, and did so silently.
        """
        depth = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        with patch.object(config, "MESH_PROVIDER_ORDER", ["depth"]), \
             patch("app.mesh_gen._estimate_depth_local", return_value=depth), \
             patch.object(config, "RELIEF_GRID", 48):
            result = generate_mesh(str(subject_image), tmp_path / "mesh")

        assert result.method == "depth", result.attempts
        assert result.status == "ok"
        assert Path(result.mesh_path).exists()

    def test_space_success_short_circuits(self, subject_image, tmp_path):
        fake = tmp_path / "fake.glb"
        fake.write_bytes(b"glTF" + b"\x00" * 600)

        with patch.object(config, "MESH_PROVIDER_ORDER", ["space", "relief"]), \
             patch("app.mesh_gen.generate_mesh_via_space",
                   return_value=(fake, "frogleo/Image-to-3D", False)):
            result = generate_mesh(str(subject_image), tmp_path / "mesh")

        assert result.method == "space"
        assert result.provider_used == "frogleo/Image-to-3D"

    def test_space_calls_are_bounded_by_the_budget(self, subject_image, tmp_path):
        """
        predict() waits as long as a Space's queue does. Each job must be given
        no more than what is left of MESH_SPACE_TIMEOUT, and cancelled on expiry.
        """
        from concurrent.futures import TimeoutError as FutureTimeout

        timeouts: list[float] = []
        jobs: list[MagicMock] = []

        def submit(api_name, **_kwargs):
            job = MagicMock()

            def result(timeout=None):
                timeouts.append(timeout)
                raise FutureTimeout()

            job.result.side_effect = result
            jobs.append(job)
            return job

        client = MagicMock()
        client.submit.side_effect = submit
        fake_gradio = MagicMock()
        fake_gradio.handle_file = lambda p: p

        with patch.dict(sys.modules, {"gradio_client": fake_gradio}), \
             patch.object(config, "MESH_SPACES", ["frogleo/Image-to-3D", "tencent/Hunyuan3D-2.1"]), \
             patch.object(config, "MESH_SPACE_TIMEOUT", 5), \
             patch("app.mesh_gen.make_gradio_client", return_value=client):
            from app.mesh_gen import generate_mesh_via_space

            with pytest.raises(RuntimeError):
                generate_mesh_via_space(str(subject_image), tmp_path / "mesh")

        assert timeouts, "no job was waited on"
        assert all(t is not None and t <= 5 for t in timeouts)
        assert all(job.cancel.called for job in jobs)

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

    def test_paths_handed_to_blender_are_absolute(self, tmp_path, monkeypatch):
        """
        Blender resolves a relative path against the .blend it is writing, not
        the working directory. Passing relative paths through produced a file
        with no texture, nothing packed, and the preview saved off in the void -
        all without a single error.
        """
        mesh = tmp_path / "m.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        Image.new("RGB", (8, 8)).save(tmp_path / "ref.png")

        seen: list[list[str]] = []

        def fake_run(cmd, log_path):
            seen.append(cmd)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            return 1, ""

        monkeypatch.chdir(tmp_path)
        with patch("app.blend_builder.resolve_blender_path", return_value=None), \
             patch("app.blend_builder._run", fake_run), \
             patch.object(config, "BLEND_WORKER_URL", ""):
            build_blend(
                "m.obj", "out",
                scene_json=normalise_scene({}, "x"),
                image_path="ref.png",
                make_preview=True,
            )

        assert seen, "no runtime was attempted"
        cmd = seen[0]
        for flag in ("--mesh", "--output", "--image", "--preview", "--glb-out", "--scene-json"):
            assert flag in cmd, f"{flag} was not passed"
            value = cmd[cmd.index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} was relative: {value}"

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

    def test_a_bpy_probe_timeout_is_not_cached(self):
        """A cold container timing out says nothing about bpy; ask again later."""
        import subprocess

        from app import blend_builder

        with patch.object(blend_builder, "_BPY_AVAILABLE", None), \
             patch("app.blend_builder.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("python", 120)):
            assert blend_builder.has_bpy_module() is False
            assert blend_builder._BPY_AVAILABLE is None


class TestWorkerHandOff:
    def _obj_dir(self, tmp_path: Path) -> Path:
        mesh_dir = tmp_path / "mesh"
        mesh_dir.mkdir()
        (mesh_dir / "mesh.obj").write_text("mtllib mesh.mtl\nv 0 0 0\n")
        (mesh_dir / "mesh.mtl").write_text("newmtl m\nmap_Kd texture.png\n")
        Image.new("RGB", (4, 4)).save(mesh_dir / "texture.png")
        return mesh_dir / "mesh.obj"

    def test_an_obj_travels_with_its_material_and_texture(self, tmp_path):
        from app.blend_builder import bundle_mesh, extract_mesh_bundle

        bundle = bundle_mesh(self._obj_dir(tmp_path), tmp_path)
        assert bundle.suffix == ".zip"

        mesh = extract_mesh_bundle(bundle, tmp_path / "unpacked")
        assert mesh.name == "mesh.obj"
        assert (mesh.parent / "mesh.mtl").exists()
        assert (mesh.parent / "texture.png").exists()

    def test_a_glb_is_uploaded_as_it_is(self, tmp_path):
        from app.blend_builder import bundle_mesh

        glb = tmp_path / "mesh.glb"
        glb.write_bytes(b"glTF")
        assert bundle_mesh(glb, tmp_path) == glb

    def test_a_bundle_cannot_write_outside_its_directory(self, tmp_path):
        import zipfile

        from app.blend_builder import extract_mesh_bundle

        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escaped.obj", "v 0 0 0\n")
        with pytest.raises(ValueError):
            extract_mesh_bundle(evil, tmp_path / "out")
        assert not (tmp_path / "escaped.obj").exists()

    def test_the_worker_gets_the_texture_image_and_token(self, tmp_path):
        from app.blend_builder import build_blend_via_worker

        served = tmp_path / "served"
        served.mkdir()
        (served / "scene.blend").write_bytes(b"B" * 2048)
        (served / "preview.glb").write_bytes(b"glTF")
        Image.new("RGB", (4, 4)).save(served / "preview.png")
        ref = tmp_path / "ref.png"
        Image.new("RGB", (4, 4)).save(ref)

        client = MagicMock()
        client.src = "https://worker.hf.space"
        client.predict.return_value = (
            str(served / "scene.blend"), str(served / "preview.glb"),
            str(served / "preview.png"), "Built with bpy",
        )
        fake_gradio = MagicMock()
        fake_gradio.handle_file = lambda p: p

        out = tmp_path / "out"
        out.mkdir()
        with patch.dict(sys.modules, {"gradio_client": fake_gradio}), \
             patch("app.mesh_gen.make_gradio_client", return_value=client), \
             patch.object(config, "BLEND_WORKER_TOKEN", "s3cret"):
            result = build_blend_via_worker(
                self._obj_dir(tmp_path), out, None, image_path=str(ref), make_preview=True
            )

        sent = client.predict.call_args.kwargs
        assert sent["mesh_file"].endswith(".zip"), "OBJ went without its texture"
        assert sent["image_file"] == str(ref)
        assert sent["token"] == "s3cret"
        assert sent["make_preview"] is True
        assert result.status == "ok"
        assert Path(result.blend_path).exists()
        assert result.glb_path and Path(result.glb_path).exists()
        assert result.preview_path and Path(result.preview_path).exists()


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

    def test_the_camera_frames_the_whole_subject(self, subject_image, tmp_path):
        """
        Every bounding-box corner inside the frame, and the subject filling a
        real share of it. Fitting the bounding sphere instead of the box framed
        the box's diagonal, and left the asset at half the height of its own
        preview.
        """
        obj, _, _ = build_solid_from_image(str(subject_image), tmp_path / "mesh", grid=48)
        result = build_blend(
            obj, tmp_path / "out",
            scene_json=normalise_scene({}, "test"),
            make_preview=False,
        )
        assert result.status == "ok", result.error_message

        probe = tmp_path / "framing.py"
        probe.write_text(
            "import bpy, json, sys\n"
            "from mathutils import Vector\n"
            "from bpy_extras.object_utils import world_to_camera_view\n"
            f"bpy.ops.wm.open_mainfile(filepath={str(result.blend_path)!r})\n"
            "scene = bpy.context.scene\n"
            "cam = scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())\n"
            "subj = bpy.data.objects['GenFX_Subject']\n"
            "xs, ys = [], []\n"
            "for corner in subj.bound_box:\n"
            "    co = world_to_camera_view(scene, cam, subj.matrix_world @ Vector(corner))\n"
            "    xs.append(co.x); ys.append(co.y)\n"
            "print('PROBE ' + json.dumps({'min_x': min(xs), 'max_x': max(xs),\n"
            "    'min_y': min(ys), 'max_y': max(ys)}))\n"
        )

        import subprocess

        blender = resolve_blender_path()
        cmd = (
            [blender, "--background", "--factory-startup", "--python", str(probe)]
            if blender else [sys.executable, str(probe)]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE ")), None)
        assert line, f"probe produced no result:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"

        box = json.loads(line[len("PROBE "):])
        assert box["min_x"] >= 0.0 and box["max_x"] <= 1.0, f"clipped horizontally: {box}"
        assert box["min_y"] >= 0.0 and box["max_y"] <= 1.0, f"clipped vertically: {box}"
        filled = max(box["max_x"] - box["min_x"], box["max_y"] - box["min_y"])
        assert filled > 0.5, f"subject fills only {filled:.0%} of the frame"

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

    def test_a_placeholder_image_is_never_meshed(self, tmp_path):
        """
        Meshing the fallback placeholder produced a solid "FALLBACK IMAGE" sign
        and reported the run as a success.
        """
        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []), \
             patch("app.pipeline.generate_mesh") as mesh, \
             patch("app.pipeline.build_blend") as blend:
            result = run_pipeline("a brass telescope")

        mesh.assert_not_called()
        blend.assert_not_called()
        assert result.ok is False
        assert "no reference image" in result.diagnostics["mesh"]
        assert "no reference image" in result.diagnostics["blend"]

    def test_the_gallery_can_be_limited_to_ones_own_runs(self, tmp_path):
        with patch.object(config, "RUNS_DIR", tmp_path / "runs"), \
             patch.object(config, "OLLAMA_ENABLED", False), \
             patch.object(config, "OPENROUTER_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "HUGGINGFACE_API_KEY", ""), \
             patch.object(config, "IMAGE_PROVIDER_ORDER", []):
            mine = run_pipeline("a mug")
            run_pipeline("someone else's lamp")
            listed = list_runs(limit=5, run_ids={mine.run_id})

        assert [entry["run_id"] for entry in listed] == [mine.run_id]

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

    def test_llm_probe_follows_the_cascade_past_a_bad_key(self):
        """A rejected OpenRouter key is a fallback, not a dead end, if Ollama is up."""
        from app.health import check_llm

        def get(url, **_kwargs):
            resp = MagicMock()
            if "openrouter" in url:
                resp.status_code = 401
            else:
                resp.status_code = 200
                resp.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
            return resp

        with patch.object(config, "LLM_PROVIDER_ORDER", ["openrouter", "ollama"]), \
             patch.object(config, "OPENROUTER_API_KEY", "sk-or-bad"), \
             patch.object(config, "OLLAMA_ENABLED", True), \
             patch.object(config, "OLLAMA_MODEL", "llama3.2"), \
             patch("app.health.requests.get", side_effect=get):
            probe = check_llm()

        assert probe["ok"] is True
        assert "Ollama" in probe["detail"]
        assert "OpenRouter HTTP 401" in probe["detail"]

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
