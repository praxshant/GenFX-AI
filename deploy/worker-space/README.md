---
title: GenFX blend worker
emoji: 🔧
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Converts a mesh into an editable Blender .blend file
---

# GenFX blend worker

The heavy half of [GenFX](https://github.com/praxshant/GenFX-Lite), deployable
on its own. It takes a mesh (or an image) and returns an editable `.blend` with
materials, a three-point light rig, a framed camera and packed textures.

Point a GenFX front end at it:

    GENFX_BLEND_WORKER=https://<your-name>-genfx-blend-worker.hf.space

Useful when the front end runs somewhere too small or too new (Python-version
wise) to host Blender itself — Streamlit Community Cloud, a 512 MB free dyno,
or any host without the pip `bpy` wheel for its Python version.

## Deploy

Copy the repository into the Space and use `deploy/worker-space/Dockerfile`
as the Space's Dockerfile.
