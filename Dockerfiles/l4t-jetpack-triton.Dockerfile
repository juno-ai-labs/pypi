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

# Ensure Triton builds the NVIDIA backend for Jetson and uses system CUDA tools
# - Limit backends to NVIDIA and disable Werror in LLVM via CMake args
# - Explicitly disable Proton (profiler) to reduce dependencies on Jetson
# - Point ptxas to the system CUDA (JetPack)
# - Provide CUDA headers to avoid online fetch of cudacrt/cudart
# - Target Ampere SM 87 via PTXAS_OPTIONS for Jetson Orin
ENV TRITON_APPEND_CMAKE_ARGS="-DTRITON_CODEGEN_BACKENDS=nvidia -DLLVM_ENABLE_WERROR=OFF -DTRITON_BUILD_PROTON=OFF" \
    TRITON_BUILD_PROTON=OFF \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUDART_PATH=/usr/local/cuda/include \
    PTXAS_OPTIONS="--gpu-name=sm_87"

RUN mkdir -p /opt /wheels

# Clone triton source
RUN git clone --branch "${TRITON_BRANCH}" --depth=1 --recursive \
    https://github.com/triton-lang/triton /opt/triton

WORKDIR /opt/triton

RUN sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    CMakeLists.txt

RUN sed -i 's|^download_and_copy_ptxas|#&|' python/setup.py || :

RUN mkdir -p third_party/cuda
RUN ln -sf /usr/local/cuda/bin/ptxas $(pwd)/third_party/cuda/ptxas

# Build triton wheel
RUN uv build --wheel --out-dir /wheels

# Test the installation
RUN uv pip install /wheels/triton*.whl && \
    python3 -c 'import triton, torch; print("Triton:", triton.__version__); print("Torch CUDA capability:", torch.cuda.get_device_capability() if torch.cuda.is_available() else "N/A")' && \
    uv pip show triton

# --- Artifact output ---
FROM scratch AS artifact
COPY --from=triton-build /wheels /wheels
