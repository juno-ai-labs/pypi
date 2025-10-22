# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE} AS base

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Shared build arguments
ARG CUDA_PYTHON_VERSION=12.6.0
ARG CUDA_PYTHON_BUILD_NUMBER=1

# Base dependencies for building cuda-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake git ninja-build \
    libcublas-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "cmake<4" scikit-build ninja cython

FROM base AS cuda-python-build

ARG CUDA_PYTHON_VERSION
ARG CUDA_PYTHON_BUILD_NUMBER

RUN mkdir -p /workspace /wheels

# Clone cuda-python source
RUN git clone --branch "v${CUDA_PYTHON_VERSION}" --depth=1 https://github.com/NVIDIA/cuda-python /workspace/cuda-python \
    || (git clone --depth=1 https://github.com/NVIDIA/cuda-python /workspace/cuda-python && cd /workspace/cuda-python && git checkout "v${CUDA_PYTHON_VERSION}")

# Install build requirements
RUN cd /workspace/cuda-python && \
    python3 -m pip install --no-cache-dir -r requirements.txt || true

ENV CUDA_PYTHON_BUILD_NUMBER=${CUDA_PYTHON_BUILD_NUMBER}

# Build cuda-python wheel
RUN cd /workspace/cuda-python && \
    export MAX_JOBS=$(nproc) && \
    python3 setup.py bdist_wheel --dist-dir /wheels

# Install cuda-python to ensure the wheel is valid and clean up
RUN python3 -m pip install --no-cache-dir /wheels/cuda_python*.whl && \
    rm -rf /workspace/cuda-python

FROM scratch AS artifact
COPY --from=cuda-python-build /wheels /wheels
