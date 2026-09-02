# Changelog

## 2.1.0

A correctness pass over the geometry and the Blender hand-off. Nothing here
changes what GenFX does; it changes whether the file it hands you is actually
the file the README describes.

Each of these produced a `.blend` that opened without complaint. That is what
made them worth hunting: the failure mode was a slightly wrong asset, not an
error message.

### The deliverable

- **Meshes were inside out.** Front and back surfaces were both wound the wrong
  way, so every normal pointed into the solid. The file opened and rendered, but
  the three-point rig was lighting the back of every face. Signed volume is now
  asserted positive in the tests.
- **Meshes were not watertight**, despite the README's word for them. The rim
  stitched only two of the several silhouette configurations, leaving a few
  hundred open edges on a typical subject. The rim is now derived from the front
  surface's unpaired directed edges, which closes it by construction — and
  orients it consistently at the same time.
- **Stray vertices are gone.** Vertices are emitted per surface cell rather than
  per mask pixel, so mask pixels too isolated to carry a face no longer arrive in
  Blender as loose points.
- **Corner-touching cells are removed.** Two quads meeting at a single point made
  that point non-manifold, which solidify, boolean, remesh and 3D printing all
  handle badly.
- **The rim samples its texture from inside the silhouette.** It used to sample
  exactly on the boundary, which is the plain background the subject was
  generated against — so every mesh wore a white edge.

### Paths, textures, preview

- **Every path handed to Blender is now absolute.** Blender resolves a relative
  path against the `.blend` it is writing, not the working directory. With a
  relative runs directory — which `.env.example` itself suggested — the texture
  was never found, `pack_all` had nothing to pack, and the preview render was
  written somewhere nobody looks. Exit code zero, no warning. CI now builds from
  a relative directory deliberately and reopens the result to prove the textures
  are packed.
- **`preview.glb` contains the subject alone.** It used to export the whole
  scene, ground plane included — and that plane is fourteen times the subject's
  extent, so a viewer that auto-frames the model showed a floor with a speck on
  it.
- **glTF imports are un-parented before centring.** They arrive under a root
  empty carrying the Y-up conversion; `location` is parent-relative, so the
  centring landed in the wrong place on exactly the highest-quality path.
- **Silhouettes stay crisp.** Blender 4.1+ auto-smooth by angle replaces blanket
  smooth shading, which was melting hard edges.
- **The ground plane sits a hair below zero**, so a flat-bottomed mesh no longer
  z-fights against it.
- **The camera frames the bounding box, not the sphere around it.** Fitting the
  sphere meant framing the box's diagonal — for a 2 m subject, 3.4 m of framing —
  and the asset sat at roughly half the height of its own preview.

### Segmentation

- **The backdrop is modelled as a gradient, not one colour.** Text-to-image
  models produce studio backdrops that fall off from dark to warm; measured
  against a single median colour, most of the backdrop read as subject. On a
  synthetic gradient the old code called 82% of the frame "subject" — the mesh
  came out as a slab with the object embedded in it. The four border strips are
  now interpolated into a per-pixel background estimate. On a real generated
  teapot, coverage went from 58% of the frame to 12%: the teapot.
- **The centred blob wins over the largest blob.** Whatever backdrop survives
  thresholding is often bigger than the subject, and picking by size handed back
  the wall instead of the object standing in front of it. Stage 2 asks for one
  centred subject, so the centre is now the seed.

### Pipeline

- **The depth tier could only ever fail.** `local() or hf()` asked NumPy for the
  truth value of a whole depth array, which raises; the exception was caught and
  the tier silently skipped. It now runs.
- **Padding no longer blurs the subject.** Square-padding applied a smoothing
  filter to the entire canvas to soften one seam. The padding colour is also a
  median of the border rather than a corner average, so one dark corner no longer
  drags the background into a visible frame.
- **Each Blender runtime gets its own log.** A shared `blender.log` meant the
  second attempt erased the evidence from the first — the one worth reading.
- **`has_bpy_module()` is cached.** The health panel was spawning an interpreter
  to import Blender on every uncached page load.
- **The worker sweeps its scratch directories.** A long-lived Space accumulated
  every mesh it had ever built.

### Front end

- **Example buttons crashed the app.** They assigned to a widget-keyed session
  value after the text area had been created, which Streamlit rejects outright.
  They are `on_click` callbacks now.
- **The `.glb` download button no longer hands you an OBJ named `.glb`.**
- **Pipeline errors are readable.** The error was drawn and then immediately
  discarded by the rerun in the `finally` block.
- **Prompt text is HTML-escaped** before it reaches the `unsafe_allow_html`
  blocks.
- **`streamlit>=1.49`** is the real floor — the UI uses `width="stretch"`, which
  older releases reject with a `TypeError`. The pin said `>=1.40`.

### Packaging

- **The worker Dockerfile installed an unpinned gradio.** `gradio>=4.44.0` was
  unquoted, so the shell read it as a redirection and wrote the build log to a
  file named `=4.44.0`.

### Tests

68 → 79. The new ones assert the properties above rather than the presence of a
file: watertightness, winding consistency, outward normals, no loose vertices,
inset rim UVs, absolute paths at the Blender boundary, camera framing measured
by projecting the subject's corners back through the saved camera, segmentation
against a gradient backdrop, and a depth tier that survives a real depth array.

## 2.0.0

Rebuilt around a single deliverable: an editable `.blend` rather than a render.
Added the image-to-3D stage, the `bpy` runtime, the worker for split
deployments, and the keyless provider cascade.
