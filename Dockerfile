# ── Stage 1: dependency install ───────────────────────────────────────────────
# We use a Python 3.12 base image so the project requirement (>=3.12) is satisfied
# without depending on Ubuntu PPAs. GPU usage works via PyTorch CUDA wheels plus
# Docker's --gpus all (the host driver provides the kernel-side components).
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

# uv-created venvs may reference /usr/bin/python3.12. The official python image
# installs CPython under /usr/local/bin, so we provide stable symlinks.
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3.12 \
 && ln -sf /usr/local/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/local/bin/python3.12 /usr/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        # needed by unstructured / libmagic
        libmagic1 \
        # PDF text extraction (used by unstructured pdfminer path)
        poppler-utils \
        # OCR fallback for image-based PDFs
        tesseract-ocr \
        # audio decode for faster-whisper / av
        ffmpeg \
        # build tools for wheels without pre-built binaries
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv — deterministic, fast package manager
RUN curl -Lsf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build

# Copy only the dependency manifest + lock file first so Docker layer cache is
# reused on code-only changes (the heavy package download layer stays cached).
COPY pyproject.toml uv.lock ./

# Create venv and install all dependencies from lock file — no network on re-build
# if the lock file hasn't changed.
RUN uv sync --frozen --no-install-project


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3.12 \
 && ln -sf /usr/local/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/local/bin/python3.12 /usr/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from the builder stage.
# This avoids shipping build tools (gcc, python3-dev…) in the final image.
COPY --from=builder /build/.venv /app/.venv

WORKDIR /app

# Copy source code
COPY . .

# Activate venv for all subsequent RUN / CMD calls
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace model cache — mount as a named volume so models survive rebuilds
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Tell torch / ctranslate2 to use GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 7860

# chainlit run uses the port defined by CHAINLIT_PORT (default 8000);
# we set it explicitly so it matches the exposed port.
ENV CHAINLIT_PORT=7860

CMD ["python", "-m", "chainlit", "run", "src/main.py", "--host", "0.0.0.0", "--port", "7860"]
