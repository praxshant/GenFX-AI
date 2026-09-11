"""
GenFX - Streamlit front end.

Prompt in, editable .blend out, with an interactive preview of the mesh in
between so you know what you are downloading before you open Blender.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.health import check_runtime_health  # noqa: E402
from app.pipeline import list_runs, load_run, run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="GenFX - prompt to editable 3D",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGE_LABELS = {
    "scene": "Scene brief",
    "image": "Reference image",
    "mesh": "3D mesh",
    "blend": "Blender file",
}

EXAMPLES = [
    "a vintage brass steampunk pocket watch",
    "a ceramic teapot with a bamboo handle",
    "a chunky retro sneaker, white and orange",
    "a carved wooden owl figurine",
    "a matte black wireless gaming mouse",
    "a potted monstera plant in a terracotta pot",
]

# ── Styling ───────────────────────────────────────────────────────────────────
# Streamlit renders this through its markdown parser, where a blank line
# followed by an indented line becomes a code block. Collapsing the CSS to
# one non-empty, unindented line per rule is what keeps it from leaking as
# visible text on the page.
STYLES = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary:#0F0F0F; --bg-surface:#161616; --bg-elevated:#1F1F1F;
    --accent:#E8D5B7; --accent-dim:#8B7355;
    --text-primary:#F0EDE8; --text-secondary:#8A8780; --text-tertiary:#55524D;
    --border:#2A2A2A; --green:#4CAF7D; --amber:#D4A853; --red:#C0392B;
    --green-bg:#16301F; --amber-bg:#332714; --red-bg:#331717; --pending-bg:#1C1C1C;
}
html, body, [class*="css"] { font-family:'DM Mono',monospace; }
.stApp { background: var(--bg-primary); }
.main .block-container { max-width:1280px; padding:2rem 2rem 5rem; }

h1,h2,h3 { font-family:'DM Serif Display',serif !important; color:var(--text-primary) !important;
           letter-spacing:-0.02em; }
p, li, label, span, div[data-testid="stMarkdownContainer"] p { color:var(--text-secondary); }

[data-testid="stSidebar"] { background:var(--bg-surface) !important;
                            border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding:1.6rem 1.1rem; }

.genfx-title { font-family:'DM Serif Display',serif; font-size:3rem; color:var(--text-primary);
               line-height:1.05; margin:0 0 .2rem; }
.genfx-subtitle { font-size:.74rem; color:var(--text-tertiary); letter-spacing:.2em;
                  text-transform:uppercase; margin-bottom:1.4rem; }

.wordmark { font-size:1.05rem; font-weight:500; color:var(--accent);
            letter-spacing:.28em; text-transform:uppercase; }
.wordmark-sub { font-size:.68rem; color:var(--text-tertiary); letter-spacing:.1em; margin-top:2px; }
.side-label { font-size:.64rem; color:var(--text-tertiary); letter-spacing:.16em;
              text-transform:uppercase; margin:1.1rem 0 .5rem; }
.side-rule { border:none; border-top:1px solid var(--border); margin:1rem 0; }

.status-row { display:flex; align-items:center; gap:8px; margin-bottom:.45rem; font-size:.75rem; }
.status-row .lbl { color:var(--text-secondary); }
.status-row .val { margin-left:auto; font-size:.64rem; letter-spacing:.05em; }
.dot-ok{color:var(--green)} .dot-fb{color:var(--amber)} .dot-pend{color:var(--text-tertiary)}
.dot-run{color:var(--accent); animation:pulse 1.1s ease-in-out infinite}
.val-ok{color:var(--green)} .val-fb{color:var(--amber)} .val-pend{color:var(--text-tertiary)}
.val-run{color:var(--accent)}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.side-detail { font-size:.62rem; color:var(--text-tertiary); margin:-.3rem 0 .6rem 20px;
               line-height:1.4; word-break:break-word; }

.flow { display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.flow-node { background:var(--bg-surface); border:1px solid var(--border); border-radius:5px;
             padding:5px 11px; font-size:.68rem; color:var(--text-secondary);
             letter-spacing:.08em; text-transform:uppercase; }
.flow-node.on { border-color:var(--accent-dim); color:var(--accent); }
.flow-arrow { color:var(--text-tertiary); font-size:.8rem; }

.card { background:var(--bg-surface); border:1px solid var(--border); border-radius:10px;
        padding:16px 18px; margin-bottom:.9rem; }
.card-head { display:flex; align-items:center; justify-content:space-between; gap:8px;
             flex-wrap:wrap; margin-bottom:.2rem; }
.card-title { font-family:'DM Serif Display',serif; font-size:1.1rem; color:var(--text-primary); }
.card-num { font-size:.62rem; color:var(--text-tertiary); letter-spacing:.16em;
            text-transform:uppercase; }
.card-note { font-size:.68rem; color:var(--text-tertiary); margin-top:.5rem; line-height:1.5; }

.badge { display:inline-flex; align-items:center; gap:5px; padding:2px 9px; border-radius:4px;
         font-size:.66rem; letter-spacing:.06em; font-weight:500; }
.badge-ok{background:var(--green-bg);color:var(--green)}
.badge-fallback{background:var(--amber-bg);color:var(--amber)}
.badge-pending{background:var(--pending-bg);color:var(--text-tertiary)}
.badge-running{background:var(--pending-bg);color:var(--accent)}

.hero { background:linear-gradient(180deg,var(--bg-surface),var(--bg-primary));
        border:1px solid var(--border); border-radius:12px; padding:20px 22px; margin-bottom:1rem; }
.hero-kicker { font-size:.64rem; color:var(--accent-dim); letter-spacing:.18em;
               text-transform:uppercase; }
.hero-title { font-family:'DM Serif Display',serif; font-size:1.5rem; color:var(--text-primary);
              margin:.25rem 0 .5rem; }
.hero-meta { font-size:.7rem; color:var(--text-tertiary); line-height:1.7; }

.stButton>button, .stDownloadButton>button {
    font-family:'DM Mono',monospace; border-radius:6px; border:1px solid var(--border);
    background:var(--bg-elevated); color:var(--text-primary); font-size:.78rem;
    letter-spacing:.04em; transition:all .14s ease; }
.stButton>button:hover, .stDownloadButton>button:hover {
    border-color:var(--accent-dim); color:var(--accent); }
.stButton>button[kind="primary"] { background:var(--accent); color:#141414;
                                   border-color:var(--accent); font-weight:500; }
.stButton>button[kind="primary"]:hover { background:#f2e2c9; color:#141414; }
.stTextArea textarea { background:var(--bg-surface) !important; color:var(--text-primary) !important;
                       border:1px solid var(--border) !important; font-family:'DM Mono',monospace !important; }
.diag { font-size:.7rem; color:var(--text-secondary); background:var(--bg-elevated);
        border-left:2px solid var(--amber); padding:7px 11px; margin-bottom:6px;
        border-radius:0 4px 4px 0; word-break:break-word; }
.empty { text-align:center; padding:3.5rem 0; }
.empty p { font-size:.8rem; color:#33312E; letter-spacing:.12em; text-transform:uppercase; }
</style>
    """


def inject_css(css: str) -> None:
    compact = "\n".join(line.strip() for line in css.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


inject_css(STYLES)


# ── Small helpers ─────────────────────────────────────────────────────────────

def esc(value: object) -> str:
    """
    Escape anything bound for the st.markdown(unsafe_allow_html=True) blocks.

    Prompts, provider names and diagnostic text all reach those blocks, and a
    prompt is free text - "a <script> prop" would otherwise be injected into
    the page rather than shown on it.
    """
    return html.escape(str(value), quote=True)


def badge(status: str) -> str:
    label = {"ok": "OK", "fallback": "FALLBACK", "running": "WORKING", "pending": "PENDING"}.get(
        status, status.upper()
    )
    dot = "○" if status == "pending" else "●"
    return f'<span class="badge badge-{status}">{dot} {label}</span>'


def status_row(label: str, status: str, detail: str = "") -> str:
    dot_cls = {"ok": "dot-ok", "fallback": "dot-fb", "running": "dot-run"}.get(status, "dot-pend")
    val_cls = {"ok": "val-ok", "fallback": "val-fb", "running": "val-run"}.get(status, "val-pend")
    dot = "○" if status == "pending" else "●"
    markup = (
        f'<div class="status-row"><span class="{dot_cls}">{dot}</span>'
        f'<span class="lbl">{esc(label)}</span>'
        f'<span class="val {val_cls}">{status.upper()}</span></div>'
    )
    if detail:
        markup += f'<div class="side-detail">{esc(detail)}</div>'
    return markup


def flow_html(active: str | None = None) -> str:
    nodes = ["Prompt", "Scene brief", "Image", "Mesh", "Blender file"]
    keys = ["prompt", "scene", "image", "mesh", "blend"]
    parts = []
    for i, (node, key) in enumerate(zip(nodes, keys)):
        cls = "flow-node on" if active and key == active else "flow-node"
        parts.append(f'<div class="{cls}">{node}</div>')
        if i < len(nodes) - 1:
            parts.append('<span class="flow-arrow">→</span>')
    return f'<div class="flow">{"".join(parts)}</div>'


@st.cache_data(ttl=300, show_spinner=False)
def cached_health() -> dict:
    return check_runtime_health()


@st.cache_data(show_spinner=False)
def file_bytes(path: str, _mtime: float) -> bytes:
    return Path(path).read_bytes()


def read_file(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return file_bytes(str(p), p.stat().st_mtime)
    except Exception:
        return None


def model_viewer(glb_path: str, height: int = 420) -> None:
    """Interactive turntable of the generated mesh, inlined as a data URI."""
    data = read_file(glb_path)
    if not data:
        st.info("No 3D preview available for this run.")
        return
    if len(data) > 24 * 1024 * 1024:
        st.info("Mesh is too large to preview in the browser - download the .blend instead.")
        return

    b64 = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
        <script type="module"
          src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
        <style>
          body {{ margin:0; background:#161616; }}
          model-viewer {{ width:100%; height:{height}px; background:#161616;
                          --poster-color:transparent; border-radius:10px; }}
          .hint {{ font-family:'DM Mono',ui-monospace,monospace; font-size:10px; color:#55524D;
                   text-align:center; letter-spacing:.14em; text-transform:uppercase;
                   padding-top:6px; }}
        </style>
        <model-viewer id="mv" src="data:model/gltf-binary;base64,{b64}"
            camera-controls auto-rotate touch-action="pan-y"
            shadow-intensity="1" exposure="1.1"
            environment-image="neutral" ar-status="not-presenting"></model-viewer>
        <div class="hint" id="hint">Drag to orbit · scroll to zoom</div>
        <script>
          // The viewer is a CDN module. If it never registers, say so rather
          // than leaving an unexplained black rectangle on the page.
          setTimeout(function () {{
            if (!window.customElements || !customElements.get('model-viewer')) {{
              document.getElementById('mv').style.display = 'none';
              document.getElementById('hint').textContent =
                'Interactive viewer unavailable - see the render below';
            }}
          }}, 6000);
        </script>
        """,
        height=height + 30,
    )


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in (
    ("result", None),
    ("running", False),
    ("live_status", {s: "pending" for s in STAGE_LABELS}),
    ("prompt_input", ""),
    ("pending_prompt", ""),
    ("fatal_error", None),
    ("my_runs", []),
):
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="wordmark">GenFX</div>'
        '<div class="wordmark-sub">prompt → editable 3D</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)

    st.markdown('<div class="side-label">Pipeline</div>', unsafe_allow_html=True)
    live = st.session_state.live_status
    st.markdown(
        "".join(status_row(STAGE_LABELS[s], live.get(s, "pending")) for s in STAGE_LABELS),
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)

    health = cached_health()
    rows = [
        ("Prompt parser", "llm"),
        ("Image model", "image"),
        ("3D model", "mesh"),
        ("Blender", "blend"),
        ("Assets", "assets"),
    ]
    st.markdown(
        "".join(
            status_row(
                label,
                "ok" if health.get(key, {}).get("ok") else "fallback",
                health.get(key, {}).get("detail") or health.get(key, {}).get("error") or "",
            )
            for label, key in rows
        ),
        unsafe_allow_html=True,
    )
    if st.button("Re-check", width="stretch"):
        cached_health.clear()
        st.rerun()

    st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
    # Only this visitor's runs, unless the operator opted into a shared gallery.
    recent = list_runs(
        limit=8,
        run_ids=None if config.SHARED_GALLERY else set(st.session_state.my_runs),
    )
    if not recent:
        st.markdown(
            '<div class="side-detail" style="margin-left:0">No runs yet.</div>',
            unsafe_allow_html=True,
        )
    for entry in recent:
        label = (entry["prompt"] or entry["run_id"])[:32]
        if st.button(f"↺ {label}", key=f"load_{entry['run_id']}", width="stretch"):
            loaded = load_run(entry["run_id"])
            if loaded:
                st.session_state.result = loaded
                st.session_state.live_status = loaded.status
                st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="genfx-title">GenFX</div>'
    '<div class="genfx-subtitle">Describe an object · get a .blend you can edit</div>',
    unsafe_allow_html=True,
)

active_stage = next(
    (s for s in STAGE_LABELS if st.session_state.live_status.get(s) == "running"), None
)
st.markdown(flow_html(active_stage), unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Describe the object you want",
    placeholder="a vintage brass steampunk pocket watch",
    height=90,
    disabled=st.session_state.running,
    key="prompt_input",
)

cols = st.columns([1.4, 1, 3.2])
with cols[0]:
    go = st.button(
        "Generate 3D  →", type="primary", width="stretch", disabled=st.session_state.running
    )
with cols[1]:
    fast = st.toggle("Fast", value=False, help="Skip the rendered preview image. Saves ~5s.")

def use_example(text: str) -> None:
    """
    Fill the prompt box from an example button.

    This has to be an on_click callback. Assigning to a widget-keyed session
    value inline would run *after* the text area was instantiated this pass,
    which Streamlit rejects outright; callbacks run before the next pass builds
    its widgets, so the assignment lands.
    """
    st.session_state.prompt_input = text


with st.expander("Try an example"):
    ex_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        ex_cols[i % 3].button(
            example, key=f"ex_{i}", width="stretch",
            on_click=use_example, args=(example,),
            disabled=st.session_state.running,
        )

if go:
    if prompt.strip():
        st.session_state.pending_prompt = prompt.strip()
        st.session_state.running = True
        st.session_state.live_status = {s: "pending" for s in STAGE_LABELS}
        st.rerun()
    else:
        st.warning("Describe an object first - a few words is enough.")

# ── Execution ─────────────────────────────────────────────────────────────────
if st.session_state.running:
    placeholder = st.empty()
    progress = st.progress(0.0)
    order = list(STAGE_LABELS)

    def on_stage(stage: str, state: str) -> None:
        st.session_state.live_status[stage] = state
        done = sum(1 for s in order if st.session_state.live_status.get(s) in ("ok", "fallback"))
        progress.progress(min(1.0, done / len(order)))
        placeholder.markdown(
            f'<div class="card"><div class="card-num">Working</div>'
            f'<div class="card-title">{esc(STAGE_LABELS.get(stage, stage))} · {esc(state)}</div></div>',
            unsafe_allow_html=True,
        )

    try:
        result = run_pipeline(
            st.session_state.pending_prompt,
            on_stage=on_stage,
            make_preview=not fast,
        )
        st.session_state.result = result
        st.session_state.my_runs.append(result.run_id)
        st.session_state.live_status = result.status
        st.session_state.fatal_error = None
    except Exception as exc:  # the pipeline swallows its own errors; this is belt-and-braces
        # Stash it: the rerun below discards anything drawn on this pass, so
        # rendering the error here would flash it away before it can be read.
        st.session_state.fatal_error = f"{type(exc).__name__} - {exc}"
    finally:
        st.session_state.running = False
        progress.empty()
        placeholder.empty()
        st.rerun()

if st.session_state.fatal_error:
    st.error(f"Pipeline error: {st.session_state.fatal_error}")

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    st.markdown(
        '<div class="empty"><p>Describe an object and press Generate 3D</p></div>',
        unsafe_allow_html=True,
    )
else:
    blend_bytes = read_file(result.blend_path)
    stats = result.blend_stats or {}

    st.markdown(
        f'<div class="hero">'
        f'  <div class="hero-kicker">{result.run_id} · {result.duration:.1f}s</div>'
        f'  <div class="hero-title">{esc(result.prompt)}</div>'
        f'  <div class="hero-meta">'
        f'    {stats.get("polygons", 0):,} polygons · {stats.get("vertices", 0):,} vertices'
        f'    · Blender {esc(stats.get("blender_version", "-"))}'
        f'    · mesh via {esc(result.providers.get("mesh") or "n/a")}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if blend_bytes:
        dl_cols = st.columns([1.5, 1, 1, 1])
        dl_cols[0].download_button(
            "⬇  Download .blend",
            data=blend_bytes,
            file_name=f"genfx_{result.run_id}.blend",
            mime="application/octet-stream",
            type="primary",
            width="stretch",
        )
        # Name the file after what it actually is. Falling back to the raw mesh
        # while still calling it .glb hands the artist an OBJ that no glTF
        # viewer will open.
        mesh_source = result.glb_path if read_file(result.glb_path) else result.mesh_path
        mesh_data = read_file(mesh_source)
        if mesh_data:
            suffix = Path(mesh_source).suffix.lower() or ".glb"
            mime = "model/gltf-binary" if suffix == ".glb" else "application/octet-stream"
            dl_cols[1].download_button(
                f"⬇  {suffix}", data=mesh_data,
                file_name=f"genfx_{result.run_id}{suffix}",
                mime=mime, width="stretch",
            )
        img = read_file(result.image_path)
        if img:
            dl_cols[2].download_button(
                "⬇  Image", data=img, file_name=f"genfx_{result.run_id}.png",
                mime="image/png", width="stretch",
            )
        if result.scene_json:
            dl_cols[3].download_button(
                "⬇  Scene JSON", data=json.dumps(result.scene_json, indent=2),
                file_name=f"genfx_{result.run_id}.json", mime="application/json",
                width="stretch",
            )
    else:
        # Name the first stage that gave up rather than guessing at a cause. A
        # degraded scene brief never stops a build, so it is not a candidate.
        failed = next(
            (s for s in ("image", "mesh", "blend")
             if result.status.get(s) != "ok" and result.diagnostics.get(s)),
            None,
        )
        reason = (
            f"{STAGE_LABELS[failed]}: {result.diagnostics[failed]}" if failed
            else "see Diagnostics below"
        )
        st.error(f"No .blend was produced - {reason}")

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="card"><div class="card-head">'
            f'<div><div class="card-num">Stage 03 · 04</div>'
            f'<div class="card-title">3D result</div></div>'
            f'{badge(result.status.get("blend", "pending"))}</div></div>',
            unsafe_allow_html=True,
        )
        preview_source = result.glb_path or result.mesh_path
        if preview_source and Path(preview_source).suffix.lower() in (".glb", ".gltf"):
            model_viewer(preview_source, height=430)
        preview_png = read_file(result.preview_path)
        if preview_png:
            st.image(preview_png, caption="Rendered from the saved .blend", width="stretch")

    with right:
        st.markdown(
            f'<div class="card"><div class="card-head">'
            f'<div><div class="card-num">Stage 02</div>'
            f'<div class="card-title">Reference image</div></div>'
            f'{badge(result.status.get("image", "pending"))}</div></div>',
            unsafe_allow_html=True,
        )
        img = read_file(result.image_path)
        if img:
            st.image(img, width="stretch")

        st.markdown(
            f'<div class="card"><div class="card-head">'
            f'<div><div class="card-num">Stage 01</div>'
            f'<div class="card-title">Scene brief</div></div>'
            f'{badge(result.status.get("scene", "pending"))}</div>'
            f'<div class="card-note">Drives the image prompt, and the material, '
            f'lighting and camera inside the .blend.</div></div>',
            unsafe_allow_html=True,
        )
        if result.scene_json:
            with st.expander("Scene JSON"):
                st.json(result.scene_json)

    st.markdown(
        '<div class="card"><div class="card-num">What you get</div>'
        '<div class="card-note">'
        'The .blend opens with the mesh centred on the origin and resting on the ground, '
        'smooth-shaded with a real material, a three-point light rig, and a framed camera '
        'on a track-to constraint. Textures are packed inside the file, so it is portable. '
        'Select the object and press Tab to start editing.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    fallbacks = [s for s, v in result.status.items() if v == "fallback"]
    with st.expander(f"Diagnostics{f' · {len(fallbacks)} stage(s) degraded' if fallbacks else ''}"):
        st.markdown(
            f'<div class="card-note">Timings: '
            + " · ".join(f"{STAGE_LABELS[s]} {result.timings.get(s, 0):.1f}s" for s in STAGE_LABELS)
            + "</div>",
            unsafe_allow_html=True,
        )
        for stage in STAGE_LABELS:
            attempts = result.attempts.get(stage) or []
            diag = result.diagnostics.get(stage)
            if not attempts and not diag:
                continue
            st.markdown(f"**{STAGE_LABELS[stage]}** · `{result.providers.get(stage) or '-'}`")
            for line in attempts:
                st.markdown(f'<div class="diag">{esc(line)}</div>', unsafe_allow_html=True)
            if diag:
                st.markdown(f'<div class="diag">{esc(diag)}</div>', unsafe_allow_html=True)
        st.caption(f"Run directory: {result.run_dir}")
