# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE} AS base

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-c"]

# Shared build args
ARG TRITON_VERSION=3.4.0
ARG TRITON_BRANCH=release/3.4.x

# --- Base build dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake git ninja-build \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip uv build twine

# --- Clone and build triton ---
FROM base AS triton-build

ARG TRITON_VERSION
ARG TRITON_BRANCH

# Ensure Triton builds only the NVIDIA backend and uses system CUDA tools on Jetson
# - TRITON_APPEND_CMAKE_ARGS overrides setup.py defaults; last -D wins in CMake
# - Disable Werror to avoid failing on warnings across toolchains
# - Point ptxas to the system CUDA (JetPack) toolchain
# - Provide CUDA headers to avoid Triton trying to fetch "cudacrt/cudart" for aarch64
# - Optionally disable the profiler (Proton) to avoid CUPTI requirements in minimal builds
ENV TRITON_APPEND_CMAKE_ARGS="-DTRITON_CODEGEN_BACKENDS=nvidia -DLLVM_ENABLE_WERROR=OFF" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUDART_PATH=/usr/local/cuda/include \
    TRITON_BUILD_PROTON=OFF

RUN mkdir -p /opt /wheels

# Clone triton source
RUN git clone --branch "${TRITON_BRANCH}" --depth=1 --recursive \
    https://github.com/triton-lang/triton /opt/triton

WORKDIR /opt/triton

# Build triton wheel
RUN uv build --wheel --out-dir /wheels

# Test the installation
RUN uv pip install /wheels/triton*.whl && \
    python3 -c 'import triton; print("Triton import OK, version:", triton.__version__)' && \
    uv pip show triton

# --- Artifact output ---
FROM scratch AS artifact
COPY --from=triton-build /wheels /wheels
