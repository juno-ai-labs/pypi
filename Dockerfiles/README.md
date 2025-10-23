# Jetson Orin Package Testing Dockerfiles

This directory contains comprehensive test Dockerfiles for validating PyTorch, TorchAudio, CUDA Python, and Triton packages on the Nvidia Jetson Orin platform. Each test suite performs extensive hardware stress testing to ensure proper CUDA compilation and GPU functionality.

## Prerequisites

### Hardware Requirements
- **Nvidia Jetson Orin** (Orin Nano, Orin NX, or AGX Orin)
- Sufficient storage space for Docker images (~5-10 GB per image)
- Active cooling recommended for stress tests

## Building the wheels

### Build PyTorch & TorchAudio wheel
```bash
mkdir -p artifacts && DOCKER_BUILDKIT=1 docker build --progress plain --build-arg BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0 --build-arg PYTORCH_VERSION=2.8.0 --build-arg TORCHAUDIO_VERSION=2.8.0 --build-arg MAX_JOBS=8 --target artifact --output type=local,dest=artifacts -f Dockerfiles/l4t-jetpack-torch-torchaudio.Dockerfile
```

```bash
mkdir -p artifacts && DOCKER_BUILDKIT=1 docker build --progress plain --build-arg BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0 --build-arg PYTORCH_VERSION=2.7.0 --build-arg TORCHAUDIO_VERSION=2.7.0 --build-arg MAX_JOBS=8 --target artifact --output type=local,dest=artifacts -f Dockerfiles/l4t-jetpack-torch-torchaudio.Dockerfile
```

## Running the Tests

All tests must be run with the `--runtime=nvidia` flag (or `--gpus all`) to enable GPU access.

### Run PyTorch Test
```bash
VERSION=2.8.0 && docker build --build-arg VERSION=$VERSION -f Dockerfiles/test-torch.Dockerfile -t jetson-test-torch:$VERSION . && docker run --runtime=nvidia --rm jetson-test-torch:$VERSION
```

```bash
VERSION=2.7.0 && docker build --build-arg VERSION=$VERSION -f Dockerfiles/test-torch.Dockerfile -t jetson-test-torch:$VERSION . && docker run --runtime=nvidia --rm jetson-test-torch:$VERSION
```

### Run TorchAudio Test
```bash
TORCH_VERSION=2.8.0 VERSION=2.8.0a0+6e1c7fe && docker build --build-arg TORCH_VERSION=$TORCH_VERSION --build-arg VERSION=$VERSION -f Dockerfiles/test-torchaudio.Dockerfile -t jetson-test-torchaudio:2.8.0 . && docker run --runtime=nvidia --rm jetson-test-torchaudio:2.8.0
```

```bash
TORCH_VERSION=2.7.0 VERSION=2.7.0a0+654fee8 && docker build --build-arg TORCH_VERSION=$TORCH_VERSION --build-arg VERSION=$VERSION -f Dockerfiles/test-torchaudio.Dockerfile -t jetson-test-torchaudio:2.7.0 . && docker run --runtime=nvidia --rm jetson-test-torchaudio:2.7.0
```

### Run CUDA Python Test
```bash
VERSION=12.6.2.post1 && docker build --build-arg VERSION=$VERSION -f Dockerfiles/test-cuda-python.Dockerfile -t jetson-test-cuda-python:$VERSION . && docker run --runtime=nvidia --rm jetson-test-cuda-python:$VERSION
```

### Run Triton Test
```bash
TORCH_VERSION=2.8.0 VERSION=3.4.0 && docker build --build-arg TORCH_VERSION=$TORCH_VERSION --build-arg VERSION=$VERSION -f Dockerfiles/test-triton.Dockerfile -t jetson-test-triton:$VERSION . && docker run --runtime=nvidia --rm jetson-test-triton:$VERSION
```

## Advanced Usage

### Save Test Output to File
```bash
docker run --runtime=nvidia --rm jetson-test-torch:latest > torch_test_results.txt 2>&1
```

### Run with Custom Memory Limits
```bash
docker run --runtime=nvidia --rm --memory="4g" jetson-test-torch:latest
```

### Interactive Mode (for debugging)
```bash
docker run --runtime=nvidia --rm -it jetson-test-torch:latest /bin/bash
# Then manually run: python3 /tests/test_torch_cuda.py
```

### Run Tests with Resource Monitoring
```bash
# Terminal 1: Run the test
docker run --runtime=nvidia --rm --name test-container jetson-test-torch:latest

# Terminal 2: Monitor GPU usage
watch -n 1 nvidia-smi
```

## Interpreting Test Results

### Successful Test Run
All tests should complete with the message:
```
================================================================================
  ALL TESTS PASSED ✓
  [Package] is properly compiled with CUDA support
  Hardware acceleration is working correctly
================================================================================
```

### Failed Tests
If a test fails, you'll see:
```
================================================================================
  TEST FAILED ✗
  Error: [error message]
================================================================================
```

Review the error output to identify the issue. Common problems include:
- GPU not detected (missing nvidia runtime)
- Out of memory errors (reduce batch sizes or tensor sizes)
- Compilation errors (incompatible CUDA versions)

## Performance Expectations

### Jetson Orin Nano (8GB)
- PyTorch matmul (2048x2048): ~150-250 GFLOPS
- Memory bandwidth: ~50-60 GB/s
- Stress test: Should complete without OOM errors

### Jetson AGX Orin (32GB/64GB)
- PyTorch matmul (4096x4096): ~500-800 GFLOPS
- Memory bandwidth: ~150-200 GB/s
- Higher throughput in all tests

## Troubleshooting

### Issue: "CUDA not available"
**Solution:** Ensure you're running with `--runtime=nvidia` or `--gpus all` flag.

### Issue: "Out of memory" errors
**Solution:** 
- Reduce test sizes by modifying the test scripts
- Close other applications using GPU memory
- Use `docker run --memory="4g"` to limit container memory

### Issue: "nvidia-container-runtime not found"
**Solution:** Install and configure nvidia-container-runtime (see Prerequisites section).

### Issue: Build fails with "package not found"
**Solution:** 
- Verify the PyPI index URL is accessible
- Check your internet connection
- Ensure the base image is available

### Issue: Tests run very slowly
**Solution:**
- Verify active cooling is working
- Check for thermal throttling with `nvidia-smi`
- Ensure power mode is set to maximum performance:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```

## Monitoring GPU During Tests

Monitor GPU usage, temperature, and power during test execution:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or with more detail
watch -n 1 'nvidia-smi && tegrastats'
```
