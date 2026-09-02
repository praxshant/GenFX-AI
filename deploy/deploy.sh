#!/usr/bin/env bash
# Deploy GenFX to a Hugging Face Space.
#
#   ./deploy/deploy.sh <hf-username> [space-name] [app|worker]
#
# Needs the Hugging Face CLI, signed in once:
#   pip install -U "huggingface_hub[cli]" && hf auth login
set -euo pipefail

USER_NAME="${1:-}"
SPACE_NAME="${2:-genfx}"
KIND="${3:-app}"

if [[ -z "$USER_NAME" ]]; then
    echo "usage: $0 <hf-username> [space-name] [app|worker]" >&2
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is not installed. Run:" >&2
    echo '  pip install -U "huggingface_hub[cli]" && hf auth login' >&2
    exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not signed in to Hugging Face. Run: hf auth login" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "→ Staging $KIND for $USER_NAME/$SPACE_NAME"
if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$REPO_ROOT" archive HEAD | tar -x -C "$STAGE"
else
    cp -r "$REPO_ROOT"/{app,ui,worker,blender,assets,deploy} "$STAGE"/
    cp "$REPO_ROOT"/{requirements.txt,create_fallback_assets.py,Dockerfile} "$STAGE"/
fi

# The README front-matter is what marks this as a Docker Space on port 7860.
if [[ "$KIND" == "worker" ]]; then
    cp "$REPO_ROOT/deploy/worker-space/README.md"  "$STAGE/README.md"
    cp "$REPO_ROOT/deploy/worker-space/Dockerfile" "$STAGE/Dockerfile"
else
    cp "$REPO_ROOT/deploy/README-hf-space.md"      "$STAGE/README.md"
fi

echo "→ Creating the Space (reused if it already exists)"
hf repos create "$USER_NAME/$SPACE_NAME" --type space --sdk docker --public --exist-ok

echo "→ Uploading"
hf upload "$USER_NAME/$SPACE_NAME" "$STAGE" . --type space \
    --exclude "**/__pycache__/**" "**/*.pyc" "runs/**" \
    --commit-message "Deploy GenFX $KIND"

echo
echo "✓ https://huggingface.co/spaces/$USER_NAME/$SPACE_NAME"
echo "  First build takes 5-10 minutes (it installs the bpy wheel)."
echo "  Then add HUGGINGFACE_API_KEY as a secret to raise the 3D quota."
