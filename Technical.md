# GenFX-AI — Technical Guide

The engineering companion to the [README](README.md): how the system is built,
how to deploy it step by step, what is known to be fragile, and where it is
going. Written against version **2.2.0**.

**Contents**

1. [Architecture](#1-architecture)
2. [Repository layout](#2-repository-layout)
3. [Prerequisites](#3-prerequisites)
4. [Deployment — exact steps](#4-deployment--exact-steps)
   - [4.1 Local on Windows](#41-local-on-windows)
   - [4.2 Local on macOS / Linux](#42-local-on-macos--linux)
   - [4.3 Docker on any machine](#43-docker-on-any-machine)
   - [4.4 Hugging Face Spaces with the deploy script](#44-hugging-face-spaces-with-the-deploy-script)
   - [4.5 Hugging Face Spaces by hand](#45-hugging-face-spaces-by-hand)
   - [4.6 Split deployment: blend worker](#46-split-deployment-blend-worker)
   - [4.7 Streamlit Community Cloud](#47-streamlit-community-cloud)
   - [4.8 Checking a deployment](#48-checking-a-deployment)
   - [4.9 Updating and rolling back](#49-updating-and-rolling-back)
5. [Configuration reference](#5-configuration-reference)
6. [What a run produces](#6-what-a-run-produces)
7. [Testing and CI](#7-testing-and-ci)
8. [Troubleshooting](#8-troubleshooting)
9. [Current potential bugs and limits](#9-current-potential-bugs-and-limits)
10. [Security notes](#10-security-notes)
11. [Future work](#11-future-work)
12. [Maintenance](#12-maintenance)

---

## 1. Architecture

```
            ┌──────────────────────── Streamlit UI (ui/streamlit_app.py) ────────────────────────┐
 prompt ──▶ │  run_pipeline()  (app/pipeline.py)                                                 │
            │                                                                                     │
            │  1 scene    llm_parser.py   OpenRouter → OpenAI → HuggingFace → Ollama → heuristic  │
            │  2 image    image_gen.py    Pollinations → HuggingFace → placeholder (run fails)    │
            │  3 mesh     mesh_gen.py     HF Spaces (time budget) → depth → inflated silhouette   │
            │  4 blend    blend_builder   Blender binary → pip bpy → remote worker (worker/app.py)│
            │                  └── subprocess: blender/build_blend.py                             │
            └─────────────────────────────── runs/run_<id>/  +  manifest.json ────────────────────┘
```

Design rules that every change should keep:

- **A stage may downgrade, never stop the run.** Each stage walks a cascade and
  records every attempt; the manifest and the Diagnostics panel explain it.
- **No stage is trusted to fail loudly.** Providers raise; the stage entry
  points (`parse_prompt`, `generate_image`, `generate_mesh`, `build_blend`)
  never do.
- **Blender always runs in a subprocess.** Blender's Python can abort the whole
  process; the web app must survive it.
- **Paths handed to Blender are absolute.** Blender resolves relative paths
  against the `.blend` being written, not the working directory.
- **A placeholder is never presented as a result.** If stage 2 falls back to the
  placeholder image, stages 3 and 4 are skipped and the run is marked failed.

Stage hand-off is plain files in the run directory, so any stage can be re-run
or inspected by hand.

---

## 2. Repository layout

```
GenFX-AI/
├── app/
│   ├── config.py          Every setting, read from environment variables
│   ├── llm_parser.py      Stage 1: provider cascade, JSON extraction, normalisation
│   ├── image_gen.py       Stage 2: image providers, square padding for 3D
│   ├── mesh_gen.py        Stage 3: Space adapters, segmentation, solid construction
│   ├── blend_builder.py   Stage 4: runtime discovery, subprocess build, worker client
│   ├── pipeline.py        Orchestration, run directories, manifests, gallery
│   └── health.py          Sidebar probes (no tokens or GPU seconds spent)
├── blender/build_blend.py Runs inside Blender / bpy: import, material, lights, camera, save
├── ui/streamlit_app.py    Front end
├── worker/app.py          Optional Gradio worker for split deployments
├── deploy/                Space cards, worker Dockerfile, deploy.sh
├── tests/test_pipeline.py Offline test suite
├── assets/fallback_image.png   Placeholder shown when no image could be generated
├── .streamlit/config.toml Theme and server settings
├── Dockerfile             Main image (Python 3.11 + bpy)
├── requirements.txt       Runtime dependencies
├── requirements-dev.txt   Runtime + pytest
├── .env.example           Every setting, documented
├── README.md              Product overview
├── CHANGELOG.md           Release history
└── Technical.md           This file
```

---

## 3. Prerequisites

| What | Needed for | Notes |
|---|---|---|
| Git | Everything | <https://git-scm.com/downloads> |
| **Python 3.11** | Writing `.blend` without installing Blender | The `bpy` wheel exists only for 3.11. Other versions run the app but need Blender or a worker for stage 4. |
| Blender 4.5+ | Optional alternative to `bpy` | Any version with a CLI; set `BLENDER_PATH` if it is not on `PATH`. |
| Docker | Container deploys | Docker Desktop on Windows/macOS. |
| Hugging Face account | Spaces deploy, better 3D quota | A free account is enough. |
| Ollama | Optional local LLM | Never installed by GenFX; used only if already running. |

No API key is required for any step. Keys only improve quality or quota.

---

## 4. Deployment — exact steps

### 4.1 Local on Windows

Uses PowerShell. The commands call the virtual environment's Python directly,
so there is no activation script to run and no execution-policy change needed.

1. **Install Python 3.11.** Download "Windows installer (64-bit)" for 3.11 from
   <https://www.python.org/downloads/windows/>. In the installer, tick
   *Add python.exe to PATH* and keep *py launcher* ticked. Check it:
   ```powershell
   py -3.11 --version
   ```
   It should print `Python 3.11.x`. Other Python versions can stay installed.
2. **Get the code.**
   ```powershell
   cd $HOME\Downloads
   git clone https://github.com/praxshant/GenFX-AI.git
   cd GenFX-AI
   ```
3. **Create a virtual environment on 3.11.**
   ```powershell
   py -3.11 -m venv .venv
   ```
4. **Install dependencies** (the `bpy` wheel is large — expect a few minutes):
   ```powershell
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
5. **Check Blender-as-a-module works:**
   ```powershell
   .venv\Scripts\python.exe -c "import bpy; print(bpy.app.version_string)"
   ```
   It should print `4.5.x`. If it fails, see [Troubleshooting](#8-troubleshooting).
6. **Create the placeholder asset** (once):
   ```powershell
   .venv\Scripts\python.exe create_fallback_assets.py
   ```
7. **Optional: settings.** Copy the template and fill in only what you have:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
   For a single-user machine, `GENFX_SHARED_GALLERY=1` keeps every run in the
   sidebar. To use a local Ollama model you already have, set
   `OLLAMA_MODEL` to its exact name from `ollama list` (for example `gemma3:4b`).
8. **Start the app:**
   ```powershell
   .venv\Scripts\python.exe -m streamlit run ui\streamlit_app.py
   ```
9. **Open** <http://localhost:8501>, type *a carved wooden owl figurine*, press
   **Generate 3D**. Outputs land in `runs\run_<id>\`.
10. **Stop** with `Ctrl+C` in the terminal.

### 4.2 Local on macOS / Linux

1. Install Python 3.11 (`brew install python@3.11` on macOS; `sudo apt install
   python3.11 python3.11-venv` on Debian/Ubuntu, via the deadsnakes PPA if your
   release lacks it).
2. Linux only — the `bpy` wheel links against these even headless:
   ```bash
   sudo apt-get install -y libgl1 libglx0 libglvnd0 libegl1 libx11-6 libxau6 \
     libxdmcp6 libxext6 libxfixes3 libxi6 libxrender1 libxkbcommon0 libsm6 libice6
   ```
3. Clone, create the environment, install:
   ```bash
   git clone https://github.com/praxshant/GenFX-AI.git
   cd GenFX-AI
   python3.11 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```
4. Check `bpy`, create the asset, run:
   ```bash
   .venv/bin/python -c "import bpy; print(bpy.app.version_string)"
   .venv/bin/python create_fallback_assets.py
   cp .env.example .env        # optional
   .venv/bin/python -m streamlit run ui/streamlit_app.py
   ```
5. Open <http://localhost:8501>.

### 4.3 Docker on any machine

1. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux) and start it.
2. From the repository root, build:
   ```bash
   docker build -t genfx .
   ```
   The first build downloads the `bpy` wheel; allow 5–10 minutes.
3. Run it, keeping runs on the host so they survive the container:
   ```bash
   docker run --rm -p 7860:7860 -v "$PWD/runs:/data/runs" genfx
   ```
   In Windows PowerShell use `-v "${PWD}\runs:/data/runs"`.
4. To pass keys, add `--env-file .env` (create it from `.env.example` first).
5. Open <http://localhost:7860>. The container reports healthy once
   `/_stcore/health` answers (`docker ps` shows `healthy`).

### 4.4 Hugging Face Spaces with the deploy script

`deploy/deploy.sh` is a Bash script. On Windows run it from **Git Bash** (comes
with Git for Windows) or WSL; on macOS/Linux use any terminal.

1. **Create a Hugging Face account** at <https://huggingface.co/join>.
2. **Create a write token:** <https://huggingface.co/settings/tokens> →
   *Create new token* → type **Write** → copy it. Keep it private.
3. **Install the CLI and sign in** (it asks for the token; paste it there, not
   into any file):
   ```bash
   python -m pip install -U "huggingface_hub[cli]"
   hf auth login
   hf auth whoami        # prints your username
   ```
4. **Commit first.** The script deploys `git archive HEAD` — uncommitted edits
   are silently left out.
   ```bash
   git status            # should be clean
   ```
5. **Deploy** (replace `<username>`):
   ```bash
   ./deploy/deploy.sh <username> genfx
   ```
   It creates a public Docker Space (reusing it if it exists), swaps in the
   Space card as `README.md`, and uploads the code.
6. **Watch the build** at `https://huggingface.co/spaces/<username>/genfx` →
   *Logs*. The first build takes 5–10 minutes.
7. **Add secrets** — Space → *Settings* → *Variables and secrets* → *New secret*:
   - `HUGGINGFACE_API_KEY` — a **read** token; raises the 3D quota. Recommended.
   - `OPENROUTER_API_KEY` / `OPENAI_API_KEY` — optional, better scene briefs.

   The Space restarts after each change.
8. **Storage (optional, paid).** Free Spaces lose files on restart. With
   persistent storage attached, runs already go to `/data/runs` (the image sets
   `GENFX_RUNS_DIR=/data/runs`).
9. Run the [deployment check](#48-checking-a-deployment).

Free Spaces sleep after a period without visitors; the next visit wakes them
and takes about a minute.

### 4.5 Hugging Face Spaces by hand

For when you cannot run Bash.

1. Create the Space: <https://huggingface.co/new-space> → name `genfx` →
   SDK **Docker** → template **Blank** → hardware **CPU basic** → *Create*.
2. Clone it next to (not inside) the project, using your write token when git
   asks for a password:
   ```bash
   git clone https://huggingface.co/spaces/<username>/genfx genfx-space
   ```
3. Copy the project in, **without** the `.git` folder, `.venv`, `runs` or `.env`.
4. Replace the Space's `README.md` with `deploy/README-hf-space.md`. Its front
   matter (`sdk: docker`, `app_port: 7860`) is what tells Spaces how to run it.
5. Commit and push:
   ```bash
   cd genfx-space
   git add -A
   git commit -m "Deploy GenFX"
   git push
   ```
6. Continue from step 6 of [4.4](#44-hugging-face-spaces-with-the-deploy-script).

### 4.6 Split deployment: blend worker

Use when the front end runs somewhere that cannot host Blender (small memory,
no Python 3.11, no system libraries).

1. **Generate a shared secret** (any long random string):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. **Deploy the worker Space:**
   ```bash
   ./deploy/deploy.sh <username> genfx-blend-worker worker
   ```
   This uses `deploy/worker-space/Dockerfile` and its Space card.
3. On the worker Space: *Settings* → *Variables and secrets* → secret
   `GENFX_WORKER_TOKEN` = the value from step 1.
4. When it is running, open the worker page. It shows the runtime line
   (`bpy module: True`) and `token required: True`.
5. On the **front end** set:
   - `GENFX_BLEND_WORKER=https://<username>-genfx-blend-worker.hf.space`
   - `GENFX_WORKER_TOKEN=` the same value
6. Generate once. The Diagnostics panel shows `Blender file · worker`.

The front end still tries a local Blender or `bpy` first; the worker is the last
runtime in the cascade.

### 4.7 Streamlit Community Cloud

Community Cloud has limited memory and no Blender system libraries, so run the
Blender stage on a worker ([4.6](#46-split-deployment-blend-worker)) and keep
the front end light.

1. Deploy the worker first (4.6, steps 1–4).
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository.
3. *Create app* → *Deploy a public app from GitHub*:
   - Repository: `praxshant/GenFX-AI`
   - Branch: `main`
   - Main file path: `ui/streamlit_app.py`
4. *Advanced settings*:
   - **Python version: 3.12.** On 3.12 the `bpy` line in `requirements.txt` is
     skipped by its version marker, which keeps the build small. (On 3.11 it
     installs, but cannot load without system libraries.)
   - **Secrets** (TOML; top-level keys become environment variables):
     ```toml
     GENFX_BLEND_WORKER = "https://<username>-genfx-blend-worker.hf.space"
     GENFX_WORKER_TOKEN = "<same secret as the worker>"
     HUGGINGFACE_API_KEY = "<read token, optional>"
     ```
5. *Deploy*. Community Cloud storage is temporary; runs disappear on restart.

### 4.8 Checking a deployment

Work down this list after any deploy:

1. The page loads and the sidebar **System** section renders.
2. **Blender** shows green (`Blender`, `bpy module`, or `remote worker`). Red
   here means no `.blend` can be produced.
3. **Assets** shows green. If not, `create_fallback_assets.py` did not run.
4. Generate *a ceramic teapot with a bamboo handle*. Expect a `.blend` in
   roughly 25–60 seconds (longer when a 3D Space is queued).
5. Download the `.blend`, open it in Blender 4.5+, check: one `GenFX_Subject`
   mesh, three lights, a camera, textures present (*File → External Data →
   Report Missing Files* reports nothing).
6. Open **Diagnostics** and read which provider each stage used. Amber stages
   are working fallbacks, not errors.

### 4.9 Updating and rolling back

- **Update:** commit on `main`, push to GitHub, re-run the same deploy command.
  Docker Spaces rebuild on every upload.
- **Roll back a Space:** check out the previous release tag locally and deploy
  it again (`git checkout <tag>`, run the deploy command, `git checkout main`).
  The Space's *Files* → *History* shows what each upload changed.
- Tag releases so there is something to roll back to:
  ```bash
  git tag v2.2.0 && git push origin v2.2.0
  ```

---

## 5. Configuration reference

All settings are environment variables (or `.env` locally, secrets on hosts).
`.env.example` documents each one.

| Variable | Default | Stage | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | — | 1 | OpenRouter key |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | 1 | OpenRouter model slug |
| `OPENAI_API_KEY` | — | 1 | OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | 1 | OpenAI model |
| `HUGGINGFACE_API_KEY` / `HF_TOKEN` | — | 1, 2, 3 | HF token (LLM, images, Space quota) |
| `HF_LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | 1 | HF router model |
| `OLLAMA_ENABLED` | `1` | 1 | `0` keeps Ollama out entirely |
| `OLLAMA_HOST` | `http://localhost:11434` | 1 | Ollama daemon |
| `OLLAMA_MODEL` | `llama3.2` | 1 | Must be pulled; a tag (`x:1b`) must match exactly |
| `LLM_PROVIDER_ORDER` | `openrouter,openai,huggingface,ollama` | 1 | Cascade order |
| `GENFX_LLM_RETRIES` | `1` | 1 | Extra tries per provider (rejected keys are never retried) |
| `GENFX_API_TIMEOUT` | `90` | 1 | Seconds per LLM request |
| `POLLINATIONS_ENABLED` | `1` | 2 | Keyless image provider |
| `POLLINATIONS_MODEL` | `flux` | 2 | Pollinations model |
| `HF_IMAGE_MODELS` | FLUX.1-schnell, SD3.5-large-turbo | 2 | Tried in order |
| `IMAGE_PROVIDER_ORDER` | `pollinations,huggingface` | 2 | Cascade order |
| `GENFX_IMAGE_WIDTH` / `_HEIGHT` | `1024` | 2 | Requested size |
| `GENFX_IMAGE_RETRIES` | `2` | 2 | Extra tries per provider |
| `GENFX_IMAGE_TIMEOUT` | `120` | 2 | Seconds per image request |
| `GENFX_MESH_SPACES` | frogleo, TRELLIS, Hunyuan3D-2.1 | 3 | Spaces tried in order |
| `MESH_PROVIDER_ORDER` | `space,depth,relief` | 3 | `relief` alone = fully offline |
| `GENFX_MESH_TIMEOUT` | `300` | 3 | Total seconds for all Spaces together |
| `GENFX_MESH_TARGET_FACES` | `30000` | 3 | Face budget (frogleo) |
| `GENFX_MESH_OCTREE` / `_STEPS` | `256` / `30` | 3 | Hunyuan-style quality knobs |
| `GENFX_DEPTH_LOCAL` | `1` | 3 | Allow the transformers depth tier |
| `GENFX_DEPTH_MODEL` | Depth-Anything-V2-Small | 3 | Depth model |
| `GENFX_RELIEF_GRID` | `220` | 3 | Local mesh resolution (cost grows fast) |
| `GENFX_RELIEF_DEPTH` | `0.45` | 3 | Local mesh thickness |
| `BLENDER_PATH` | `blender` | 4 | Blender executable |
| `GENFX_BLEND_TIMEOUT` | `300` | 4 | Seconds per Blender runtime |
| `GENFX_BLEND_PREVIEW` | `1` | 4 | Render `preview.png` |
| `GENFX_PREVIEW_SAMPLES` | `16` | 4 | Preview quality |
| `GENFX_BLEND_WORKER` | — | 4 | Worker URL |
| `GENFX_WORKER_TOKEN` | — | 4 | Shared worker secret |
| `GENFX_RUNS_DIR` | `./runs` (`/data/runs` in Docker) | — | Output directory |
| `GENFX_MAX_RUNS` | `40` | — | Older runs are deleted |
| `GENFX_SHARED_GALLERY` | `0` | UI | `1` lists every run, not just this session's |

---

## 6. What a run produces

`runs/run_<8 hex>/`:

| File | Written by | Notes |
|---|---|---|
| `scene.json` | stage 1 | The normalised brief |
| `image.png` | stage 2 | Squared, centre-padded reference |
| `mesh/mesh.glb` or `mesh/mesh.obj` + `.mtl` + `texture.png` | stage 3 | GLB from a Space, OBJ from the local tiers |
| `mesh_bundle.zip` | stage 4 | Only for worker builds of an OBJ |
| `scene.blend` | stage 4 | The deliverable, textures packed |
| `preview.glb` | stage 4 | Subject only, for the web viewer |
| `preview.png` | stage 4 | Workbench render (skipped in *Fast* mode) |
| `blender-<runtime>.log` | stage 4 | One per runtime attempted |
| `manifest.json` | pipeline | Status, provider, attempts and timing per stage |

`manifest.json` is the first thing to read when a run looks wrong.

---

## 7. Testing and CI

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

- 99 tests; offline — no keys, no network, no Blender. HTTP, Spaces and the
  worker are faked.
- 3 integration tests need a Blender runtime and skip without one. With one,
  they build real `.blend` files, reopen them, and check the subject, lights,
  camera framing and packed textures.
- The mesh tests assert geometry, not just file existence: watertight,
  consistently wound, outward normals, no loose vertices.

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and on pull
requests: installs the Blender system libraries and `requirements-dev.txt` on
Python 3.11, runs the suite (Blender tests included), then builds a `.blend` from
a relative output directory and reopens it to prove textures were packed.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sidebar **Blender** red | No Blender, `bpy` unavailable, no worker | Use Python 3.11 + `requirements.txt`, set `BLENDER_PATH`, or configure a worker |
| `import bpy` fails on Linux | Missing system libraries | Install the `apt` list in [4.2](#42-local-on-macos--linux) |
| `pip` cannot find `bpy==4.5.3` | Not Python 3.11 | Recreate the venv with 3.11 |
| Run fails at **Reference image** | Pollinations rate-limited or down, no HF token | Wait and retry, or add `HUGGINGFACE_API_KEY` |
| Mesh always `relief` | Spaces out of quota, down, or over the time budget | Add an HF token; raise `GENFX_MESH_TIMEOUT`; check the Space pages |
| Scene brief always `local_heuristic` | No key works and Ollama unusable | Check keys in the sidebar; for Ollama set `OLLAMA_MODEL` to a pulled model |
| Sidebar says `Ollama has no 'llama3.2'` | Different models pulled | `ollama list`, then set `OLLAMA_MODEL` to one of them |
| Worker build: `Unauthorised` | Token mismatch | Same `GENFX_WORKER_TOKEN` on both sides |
| `.blend` opens untextured | Mesh had no texture (shape-only Space) | Expected: a PBR material from the brief is applied instead |
| No 3D viewer in the page | CDN blocked, or mesh over 24 MB | Use `preview.png` or open the `.blend` |
| Space stuck *Building* | First `bpy` install | Wait 10 minutes; read the build log |
| Recent runs vanished | Session ended, or temporary storage restarted | Set `GENFX_SHARED_GALLERY=1` locally; attach storage on hosts |

For anything else: open **Diagnostics**, then the run's `manifest.json`, then
the `blender-*.log` files.

---

## 9. Current potential bugs and limits

Known today and not yet fixed. Severity is a judgement of user impact.

### Likely to be hit

1. **Hard-coded Space APIs** — `mesh_gen._space_payload` names each Space's
   endpoint and parameters. When a Space owner changes its Gradio API, calls
   fail and every run silently drops to the local tier. The Hunyuan3D-2.1
   parameter set in particular has not been verified against the live Space.
   *Impact: high. Fix: detect with `client.view_api()`; add a scheduled smoke test.*
2. **The page blocks for the whole run.** The pipeline runs inside the Streamlit
   script. Worst case, with every timeout expiring, stages can add up to tens
   of minutes (LLM 90 s × tries × providers; images 120 s × tries × providers;
   Spaces 300 s; Blender 300 s per runtime). *Fix: background job queue (see §11).*
3. **Sidebar buttons stay live during a run.** *Re-check* and *Recent* are not
   disabled while generating; clicking one interrupts the script and leaves a
   half-written run directory with no manifest (pruned later, but the user
   loses the run).
4. **Free-tier dependencies can change terms.** Pollinations (keyless images)
   and anonymous ZeroGPU quotas are third-party and rate-limited; the default
   `OPENROUTER_MODEL` is a `:free` slug that may be retired. A retired model now
   falls through quickly, but brief quality silently drops. The sidebar checks
   the key, not the model.

### Resource and scaling

5. **Unbounded file cache in the UI.** `file_bytes` in `ui/streamlit_app.py`
   uses `@st.cache_data` with no `max_entries`, so every downloaded `.blend`,
   `.glb` and image stays in memory for the life of the server.
   *Fix: `max_entries=` / `ttl=`, or stream from disk.*
6. **Pruning can delete a run someone is viewing.** `_prune_old_runs` keeps the
   newest `GENFX_MAX_RUNS` by modification time across all users. On a busy
   deployment a visitor's result can be deleted before they download it.
7. **Pure-Python loops in segmentation.** The flood fill and distance transform
   are Python loops; fine at the default 220 grid, but cost grows with the
   square of `GENFX_RELIEF_GRID`.
8. **Large previews inflate the page.** The 3D viewer embeds the GLB as base64
   (+33 %) — up to ~32 MB of HTML for a 24 MB mesh.
9. **Depth tier downloads at runtime.** With `transformers` + `torch` installed,
   the first run downloads the depth model; not suitable for offline machines
   unless pre-cached.

### Correctness edge cases

10. **Wrong Blender picked on Windows.** `resolve_blender_path` sorts install
    folders as text, so `Blender 4.5` sorts above `Blender 4.10` and the older
    one is chosen. *Fix: sort by parsed version numbers.*
11. **A found Blender beats `bpy`.** Any `blender` on `PATH` is used first, even an
    old 3.x; features fall back but the file is then written by that version.
12. **Heuristic briefs are generic.** Without an LLM, material colour, lighting
    and camera are defaults; visible only when the mesh arrives untextured.
13. **Image prompts are cut at 1,800 characters** for the Pollinations URL.
14. **The single-subject assumption.** Segmentation picks the blob at the image
    centre; off-centre or multi-object images reconstruct poorly.
15. **Tier 3 is an inflation, not a reconstruction.** The back is a scaled
    mirror of the front.

### Operational

16. **No authentication or rate limiting on the front end.** Anyone with the
    URL can spend your API credit and HF quota.
17. **Runs are lost on restart** without persistent storage (free Spaces,
    Community Cloud).
18. **`deploy.sh` ships only committed code** (`git archive HEAD`) and needs Bash.
19. **CI downloads the `bpy` wheel on every run** (pip cache helps, but the
    wheel is large).
20. **Session-scoped gallery.** "Recent" is kept in the browser session and
    empties on reload unless `GENFX_SHARED_GALLERY=1`.

---

## 10. Security notes

- **Secrets** live only in environment variables / host secret stores. `.env` is
  git-ignored; `config.summary()` exposes booleans, never values (tested).
- **HTML injection:** every user- or provider-supplied string rendered with
  `unsafe_allow_html` goes through `esc()`.
- **Tracebacks** are hidden from visitors (`showErrorDetails = false`).
- **Worker:** set `GENFX_WORKER_TOKEN` on any worker reachable from the
  internet; comparison is constant-time. Uploads are capped at 200 MB, zip
  bundles at 512 MB uncompressed, and bundle entries with absolute paths, `..`
  or subdirectories are rejected.
- **Untrusted files:** the worker runs Blender's importers on uploaded meshes.
  Keep it isolated (its own Space/container, no secrets beyond the token).
- **Public front end:** see bug 16 — consider a password (Spaces can be made
  private) or a reverse proxy with auth and rate limits.

---

## 11. Future work

Roughly in priority order.

**Reliability**
- Background job queue (worker process + polling UI) so runs survive page
  reloads, can be cancelled, and never block the server thread.
- Scheduled smoke test against each configured Space; auto-disable a Space whose
  API changed.
- Bound the UI file cache; make pruning skip runs touched in the last hour.
- Sort Blender installs by version; prefer `bpy` 4.5 over an older binary.

**Quality**
- Background removal with a real segmentation model (rembg / SAM) before
  meshing.
- Multi-view generation (front/back/side images) for Spaces that accept them.
- Automatic decimation, retopology and UV unwrap for Space meshes that arrive
  without UVs.
- PBR map estimation (roughness/normal) from the reference image.
- Pick an installed Ollama model automatically when `OLLAMA_MODEL` is not pulled.

**Product**
- Upload an image directly (skip stages 1–2) — the worker already has this path.
- Extra exports: FBX, USDZ, STL for printing.
- Persistent per-user gallery with sign-in.
- A small REST API for batch generation.
- Seed control and "regenerate this stage only".

**Operations**
- Authentication and rate limiting for public deployments.
- Pinned dependency lockfile for reproducible builds.
- Smaller Docker image (multi-stage build, strip test files).
- Structured logs and basic metrics (run duration, tier hit rates).

---

## 12. Maintenance

- **Commit identity:** commits in this repository are authored as
  `Prashant Gagneja <prashantgagneja0@gmail.com>`.
- **Branching:** `main` is deployable. Work on a branch, open a pull request, let
  CI pass, then merge.
- **Releases:** bump `APP_VERSION` in `app/config.py`, add a `CHANGELOG.md`
  section, tag `vX.Y.Z`, push the tag, redeploy.
- **Dependencies:** `bpy` is pinned to the 4.5 LTS wheel for Python 3.11; moving
  to a newer Blender means moving Python to match, in `Dockerfile`,
  `deploy/worker-space/Dockerfile`, `requirements.txt` and CI together.
- **When a Space changes:** update `_space_payload`, add a test with the new
  response shape, and note it in the changelog.
