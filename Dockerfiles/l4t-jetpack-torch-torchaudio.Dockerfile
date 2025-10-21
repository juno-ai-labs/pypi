# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE} AS base

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Shared build arguments
ARG PYTORCH_VERSION=2.8.0
ARG PYTORCH_BUILD_NUMBER=1
ARG TORCHAUDIO_VERSION=2.8.0
ARG TORCHAUDIO_BUILD_NUMBER=1
ARG NUMPY_VERSION=1.26.4
ARG TORCH_CUDA_ARCH_LIST="8.7"
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"

# Base dependencies reused across build stages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake git ninja-build rsync \
    libopenblas-dev libomp-dev gfortran pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    libavfilter-dev libavdevice-dev libavcodec58 libavformat58 \
    libavutil56 libswresample3 libavfilter7 libavdevice58 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    USE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
    BUILD_TESTING=OFF \
    BUILD_TESTING_CPP=OFF \
    CMAKE_BUILD_TESTING=OFF

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "cmake<4" scikit-build ninja

FROM base AS pytorch-build

ARG PYTORCH_VERSION
ARG PYTORCH_BUILD_NUMBER
ARG NUMPY_VERSION

RUN mkdir -p /workspace /wheels

# Clone PyTorch source (retry with shallow clone fallback if tag missing)
RUN git clone --branch "v${PYTORCH_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /workspace/pytorch \
    || (git clone --depth=1 --recursive https://github.com/pytorch/pytorch /workspace/pytorch && cd /workspace/pytorch && git checkout "v${PYTORCH_VERSION}")

# Patch build and install requirements
RUN cd /workspace/pytorch && \
    sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' third_party/cpuinfo/src/arm/linux/aarch64-isa.c && \
    python3 -m pip install --no-cache-dir numpy==${NUMPY_VERSION} && \
    python3 -m pip install --no-cache-dir -r requirements.txt

ENV PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER} \
    USE_CUDNN=1 \
    USE_CUSPARSELT=0 \
    USE_CUDSS=0 \
    USE_CUFILE=0 \
    USE_NATIVE_ARCH=1 \
    USE_DISTRIBUTED=1 \
    USE_NCCL=1 \
    USE_GLOO=1 \
    USE_MPI=0 \
    USE_FBGEMM=0 \
    USE_NNPACK=1 \
    USE_XNNPACK=1 \
    USE_PYTORCH_QNNPACK=1 \
    USE_FLASH_ATTENTION=1 \
    USE_MEM_EFF_ATTENTION=1 \
    USE_TENSORRT=0 \
    USE_BLAS=1 \
    BLAS="OpenBLAS" \
    USE_PRIORITIZED_TEXT_FOR_LD=1 \
    BUILD_NVFUSER=1 \
    USE_GTEST=0

# Build PyTorch wheel
RUN cd /workspace/pytorch && \
    export MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
    python3 setup.py bdist_wheel --dist-dir /wheels

# Install PyTorch for subsequent stages and clean workspace
RUN python3 -m pip install --no-cache-dir /wheels/torch*.whl && \
    rm -rf /workspace/pytorch

FROM pytorch-build AS torchaudio-build

ARG TORCHAUDIO_VERSION
ARG TORCHAUDIO_BUILD_NUMBER

# Clone TorchAudio source (retry fallback)
RUN git clone --branch "v${TORCHAUDIO_VERSION}" --depth=1 --recursive https://github.com/pytorch/audio /workspace/torchaudio \
    || (git clone --depth=1 --recursive https://github.com/pytorch/audio /workspace/torchaudio && cd /workspace/torchaudio && git checkout "v${TORCHAUDIO_VERSION}")

# Install build requirements
RUN cd /workspace/torchaudio && \
    python3 -m pip install --no-cache-dir -r requirements.txt

ENV TORCHAUDIO_BUILD_NUMBER=${TORCHAUDIO_BUILD_NUMBER} \
    USE_FFMPEG=1 \
    USE_SOX=0

# Build TorchAudio wheel
RUN cd /workspace/torchaudio && \
    export MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
    python3 setup.py bdist_wheel --dist-dir /wheels

# Install TorchAudio to ensure the wheel is valid and clean up
RUN python3 -m pip install --no-cache-dir /wheels/torchaudio*.whl && \
    rm -rf /workspace/torchaudio

FROM scratch AS artifact
COPY --from=torchaudio-build /wheels /wheels

