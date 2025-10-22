# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE} AS base

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-c"]

# Shared build args
ARG CUDA_PYTHON_VERSION=12.9.4
ARG MAX_JOBS=8

# --- Base build dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip uv build twine

# --- Clone and build cuda-python (12.6+) ---
FROM base AS cuda-python-build

ARG CUDA_PYTHON_VERSION=12.9.4
ARG MAX_JOBS=8

RUN mkdir -p /workspace /wheels && \
    git clone --branch "v${CUDA_PYTHON_VERSION}" --depth=1 \
      https://github.com/NVIDIA/cuda-python /workspace/cuda-python

WORKDIR /workspace/cuda-python

# Build both wheels (new structure: cuda_core + cuda_bindings)
RUN set -ex && \
    cd cuda_core && \
    uv build --wheel . --out-dir /wheels --verbose && \
    cd ../cuda_bindings && \
    uv build --wheel . --out-dir /wheels --verbose

# --- Artifact output ---
FROM scratch AS artifact
COPY --from=cuda-python-build /wheels /wheels
