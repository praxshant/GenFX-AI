---
title: GenFX
emoji: 🧊
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Describe an object, get an editable .blend file
---

# GenFX

Type a description, get a `.blend` you can open in Blender and edit.

    prompt → scene brief → reference image → 3D mesh → editable .blend

Works with no API keys. Adding a Hugging Face token (Settings → Variables and
secrets → `HUGGINGFACE_API_KEY`) raises the ZeroGPU quota on the image-to-3D
Spaces this app calls, which makes the highest-quality mesh path available more
often.

## Recommended Space settings

| Setting  | Value                                       |
|----------|---------------------------------------------|
| SDK      | Docker                                       |
| Hardware | CPU basic (2 vCPU · 16 GB) is enough         |
| Storage  | optional — set `GENFX_RUNS_DIR=/data/runs`   |

## Optional secrets

| Name                    | Effect                                              |
|-------------------------|-----------------------------------------------------|
| `HUGGINGFACE_API_KEY`   | Higher 3D quota, textured meshes, backup image model |
| `OPENROUTER_API_KEY`    | Better scene briefs and image prompts                |
| `GENFX_BLEND_WORKER`    | Offload `.blend` building to a second Space          |

Source: https://github.com/praxshant/GenFX-Lite
