# GenFX blend worker - the .blend build stage, deployable on its own.
# Build from the repository root:
#   docker build -f deploy/worker-space/Dockerfile -t genfx-worker .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/genfx \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx0 libglvnd0 libegl1 \
        libx11-6 libxau6 libxdmcp6 libxext6 libxfixes3 libxi6 libxrender1 \
        libxkbcommon0 libsm6 libice6 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 genfx

WORKDIR /app

COPY requirements.txt .
# Quote the specifier: unquoted, the shell reads ">=4.44.0" as a redirection,
# installs whatever gradio is latest and writes the build log into a file
# called "=4.44.0".
RUN pip install --no-cache-dir -r requirements.txt "gradio>=4.44.0"

COPY --chown=genfx:genfx . .
RUN python create_fallback_assets.py && chown -R genfx:genfx /app

USER genfx

EXPOSE 7860
CMD ["python", "worker/app.py"]
