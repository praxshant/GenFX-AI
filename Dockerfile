# GenFX - prompt to editable .blend
#
# Python 3.11 is not incidental: the pip `bpy` wheel is built per Python minor
# version, and 4.5 LTS targets 3.11. Change the base image and you lose the
# ability to write .blend files without installing Blender itself.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GENFX_RUNS_DIR=/data/runs \
    HOME=/home/genfx \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp

# Exactly what `ldd bpy/__init__.so` asks for, plus libegl1 for the EEVEE
# preview render. Anything more just inflates the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglx0 libglvnd0 libegl1 \
        libx11-6 libxau6 libxdmcp6 libxext6 libxfixes3 libxi6 libxrender1 \
        libxkbcommon0 libsm6 libice6 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 genfx && mkdir -p /data/runs && chown -R genfx:genfx /data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=genfx:genfx . .

RUN python create_fallback_assets.py && chown -R genfx:genfx /app

USER genfx

EXPOSE 7860
HEALTHCHECK --interval=45s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
