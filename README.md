# GenFX

**Describe an object. Get a `.blend` you can open in Blender and edit.**

```
prompt  →  scene brief  →  reference image  →  3D mesh  →  editable .blend
```

GenFX turns a sentence into a real 3D asset: a watertight textured mesh,
centred on the world origin and resting on the ground, with a PBR material, a
three-point light rig, a framed camera, and every texture packed inside the
file. Open it, press <kbd>Tab</kbd>, and start modelling.

It runs with **no API keys at all**, and deploys on free infrastructure.

<!-- Add a screenshot at docs/screenshot.png once you have deployed. -->

---

## What it produces

| Artifact | What it is |
|---|---|
| `scene.blend` | The deliverable — mesh, material, lights, camera, packed textures |
| `preview.glb` | The same scene as glTF, for web viewers and game engines |
| `preview.png` | A render from the saved `.blend` |
| `image.png` | The reference image the mesh was reconstructed from |
| `scene.json` | The structured brief that drove every later stage |
| `manifest.json` | Which provider each stage used, with timings and diagnostics |

Every run lands in `runs/run_<id>/`.

---

## Quick start

```bash
git clone https://github.com/praxshant/GenFX-AI.git
cd GenFX-AI

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python create_fallback_assets.py    # once
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501, type *"a carved wooden owl figurine"*, press
**Generate 3D**. About 25–30 seconds later there is a `.blend` to download.

> **Python 3.11 matters.** The `bpy` wheel — Blender as a pip package — is built
> per Python minor version, and the 4.5 LTS line targets 3.11. On 3.11 you get
> `.blend` files with no Blender installed at all. On another version, point
> `BLENDER_PATH` at a real Blender, or set `GENFX_BLEND_WORKER`.

No `.env` is required. To go further, `cp .env.example .env` — every setting is
documented there.

---

## How each stage degrades

The design rule: **a stage may downgrade, but it may never stop the run.** Each
one walks a cascade and records what it tried, so the Diagnostics panel always
explains what happened.

### 1 · Prompt → scene brief

OpenRouter → OpenAI → HuggingFace → **Ollama (local)** → local heuristic.

The brief is not decoration — it is consumed downstream. `image_prompt` drives
stage 2; `materials`, `lighting` and `camera` drive the Blender scene in stage
4. With no provider configured, a deterministic local parser still writes a
usable brief from your words, so the pipeline runs offline.

Ollama is never installed or downloaded by GenFX. If the daemon is running it
is used; if not, the cascade moves on:

```bash
ollama pull llama3.2 && ollama serve      # optional, free, private
```

### 2 · Scene brief → reference image

**Pollinations (keyless)** → HuggingFace Inference → pre-baked asset.

The image is squared and centre-padded before it moves on, because image-to-3D
reconstruction is far more reliable with one centred subject on a plain
background — which is also why the system prompt asks for exactly that.

### 3 · Image → 3D mesh

If stage 2 could not produce a real image, this stage and the next are skipped
and the run is reported as failed. Reconstructing the placeholder would hand
you a solid "FALLBACK IMAGE" sign and call it a success.

1. **Hosted image-to-3D** — Hunyuan3D and TRELLIS, called through public
   Hugging Face Spaces. Real geometry: ~30k polygons in 15–25 seconds. Free;
   an HF token raises the ZeroGPU quota and unlocks textured pipelines. All
   Spaces together get `GENFX_MESH_TIMEOUT` seconds (default 300); a job
   still queued when that runs out is cancelled and the next tier takes over.
2. **Depth-displaced solid** — monocular depth (Depth Anything V2) shaping a
   textured mesh. Used when a Space is down and the optional
   `transformers` + `torch` are installed (`pip install transformers torch`).
3. **Inflated silhouette** — the subject is segmented from its background and
   inflated with a distance transform into a closed, textured solid. Pure
   NumPy, no model, no key, no network. This one always works.

Segmentation models the backdrop as a gradient rather than a single colour,
because that is what image models actually produce — a studio falloff from dark
to warm. Against one colour most of the backdrop reads as subject and the mesh
comes out as a slab with the object buried in it. The four border strips are
interpolated into a per-pixel estimate instead, and the blob covering the centre
of the frame wins over the largest blob: whatever backdrop survives thresholding
is often bigger than the subject standing in front of it.

Whichever tier produced it, the local solid is built to three invariants the
tests assert on every run, because each is invisible until it ruins something
downstream:

| Invariant | What breaks without it |
|---|---|
| Every edge borders exactly two faces | Holes; solidify, boolean and 3D printing all fail |
| Every directed edge used once | Inconsistent orientation, patchy shading |
| Positive signed volume | The solid is inside out — the light rig lands on the back of every face |

The rim also samples its texture a few pixels *inside* the silhouette. Sampled
where it sits, it would pick up the plain background the subject was generated
against, and every mesh would wear a white edge.

### 4 · Mesh → `.blend`

Blender binary → pip `bpy` module → remote worker Space.

The build always runs in a subprocess: Blender's Python can abort a process
outright, and that must not take the web app with it. Each runtime writes its
own `blender-<runtime>.log` in the run directory, so a failed first attempt
still has its evidence after the second one runs.

Every path handed to Blender is made absolute first. Blender resolves a
relative path against the `.blend` it is writing rather than the working
directory, so a relative runs directory produced a file with no texture,
nothing packed and the preview saved somewhere else entirely — at exit code
zero, with no warning. CI now builds from a relative directory on purpose and
reopens the result to check.

---

## Deploying free

Verified free-tier options, best first.

### Hugging Face Spaces — Docker *(recommended)*

CPU basic gives 2 vCPU / 16 GB / 50 GB, which is far more headroom than any
other free tier, and Docker means the `bpy` runtime is guaranteed.

```bash
pip install -U "huggingface_hub[cli]" && hf auth login
./deploy/deploy.sh <your-hf-username> genfx
```

Then in Space settings add `HUGGINGFACE_API_KEY` as a secret (optional but
recommended) and set `GENFX_RUNS_DIR=/data/runs` if you attach storage.

`deploy/README-hf-space.md` is the Space card the script installs.

### Streamlit Community Cloud

Free and simple, but memory is tight and the Python version is not always 3.11.
Deploy the front end there and offload the heavy stage:

1. Deploy the worker: `./deploy/deploy.sh <username> genfx-blend-worker worker`
2. Pick a long random secret and set it as `GENFX_WORKER_TOKEN` on the worker
   Space. Without it, anyone who finds the worker can make it run Blender.
3. In the Streamlit app's secrets, set
   `GENFX_BLEND_WORKER = "https://<username>-genfx-blend-worker.hf.space"` and
   the same `GENFX_WORKER_TOKEN`.

An OBJ mesh is zipped with its material and texture for the trip, so the
worker's `.blend` is textured, and the worker sends back the preview render
and `.glb` along with it.

This is the split ("distributed") deployment: a light front end anywhere, the
Blender stage on hardware that can carry it. The same trick works for Render,
Koyeb, Fly.io, or any 512 MB free container.

### Anywhere else, with Docker

```bash
docker build -t genfx .
docker run -p 7860:7860 -v "$PWD/runs:/data/runs" genfx
```

---

## Configuration

Everything is environment-driven; `.env.example` lists all of it. The settings
that change the most:

| Variable | Default | Why you would change it |
|---|---|---|
| `HUGGINGFACE_API_KEY` | *(none)* | The highest-value single setting: more 3D quota, textured meshes |
| `OPENROUTER_API_KEY` | *(none)* | Better scene briefs and image prompts |
| `OLLAMA_HOST` | `http://localhost:11434` | Point at a local Ollama daemon |
| `MESH_PROVIDER_ORDER` | `space,depth,relief` | Force offline meshing with `relief` |
| `GENFX_MESH_SPACES` | 3 Spaces | Swap in your own image-to-3D Space |
| `BLENDER_PATH` | `blender` | Use a real Blender install |
| `GENFX_MESH_TIMEOUT` | `300` | Total seconds the image-to-3D Spaces may take |
| `GENFX_BLEND_WORKER` | *(none)* | Split deployment |
| `GENFX_WORKER_TOKEN` | *(none)* | Shared secret between front end and worker |
| `GENFX_RUNS_DIR` | `./runs` | Persistent storage path |
| `GENFX_SHARED_GALLERY` | `0` | `1` shows every run in "Recent" — fine locally, not on a public deployment |

---

## Project layout

```
app/
  config.py         Environment-driven settings
  llm_parser.py     Stage 1 · provider cascade, JSON extraction, normalisation
  image_gen.py      Stage 2 · keyless and keyed image providers
  mesh_gen.py       Stage 3 · Space adapters + local mesh construction
  blend_builder.py  Stage 4 · runtime discovery, subprocess isolation
  pipeline.py       Orchestration, run directories, manifests
  health.py         Cheap non-destructive probes for the sidebar
blender/
  build_blend.py    Runs inside Blender or under the pip bpy module
ui/
  streamlit_app.py  The front end
worker/
  app.py            Optional Gradio worker for split deployments
deploy/             Space cards, worker Dockerfile, deploy script
tests/              99 tests, no keys, no network, no Blender required
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 99 tests; the 3 Blender integration tests skip without a runtime
```

Offline by design: no API keys, no network, no Blender. The tests that need a
Blender runtime skip themselves when there isn't one — and when there is, they
reopen the generated `.blend` and assert the mesh, the material, the three
lights, the camera and the packed textures all survived the round trip.

The mesh tests check the geometry itself rather than just that a file appeared:
watertightness, winding consistency, outward normals, no loose vertices. A mesh
can be none of those things and still open in Blender looking roughly right,
which is exactly why they are asserted rather than eyeballed.

---

## Known limits

- **Mesh quality tracks the reference image.** One clear object on a plain
  background reconstructs well; a busy scene does not. The system prompt steers
  hard toward the former.
- **Public Spaces have quotas.** Anonymous ZeroGPU runs out quickly; a free HF
  token raises it substantially. When it is gone, tier 3 still delivers a mesh.
- **Tier 3 is an inflation, not a reconstruction.** It produces a convincing
  solid from one view — the back is a mirrored shell, not real geometry.
- **Meshes are unrigged and UV-unwrapped only where the source provides it.**
  Retopology and rigging are yours to do in Blender.
- **`.blend` files are written by Blender 4.5.** Blender 4.5+ opens them
  natively; older versions may not.

## License

MIT
