# Jetson Orin Package Testing Dockerfiles

This directory contains comprehensive test Dockerfiles for validating PyTorch, TorchAudio, CUDA Python, and Triton packages on the Nvidia Jetson Orin platform. Each test suite performs extensive hardware stress testing to ensure proper CUDA compilation and GPU functionality.

## Prerequisites

### Hardware Requirements
- **Nvidia Jetson Orin** (Orin Nano, Orin NX, or AGX Orin)
- Sufficient storage space for Docker images (~5-10 GB per image)
- Active cooling recommended for stress tests

### Software Requirements
- **JetPack 6.1** (or compatible version with R36.4.0 L4T)
- **Docker Engine** with nvidia runtime support
- **nvidia-container-runtime** configured

### Setup nvidia-docker Runtime

1. Install nvidia-container-runtime (if not already installed):
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime
```

2. Configure Docker to use nvidia runtime:
```bash
sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
```

3. Restart Docker:
```bash
sudo systemctl restart docker
```

4. Verify nvidia runtime is available:
```bash
docker info | grep -i runtime
```

## Test Dockerfiles Overview

### 1. PyTorch Test (`test-torch.Dockerfile`)
Tests PyTorch's CUDA integration with comprehensive GPU operations.

**Tests Include:**
- CUDA availability and device properties
- Basic tensor operations on GPU
- GPU memory management and allocation
- CPU-GPU data transfer performance
- Matrix operations at various sizes
- Convolution operations (CNN workload)
- Mixed precision (FP16/FP32) operations
- Multi-GPU awareness
- Tensor Core operations (Ampere architecture)
- 30-second sustained GPU stress test

### 2. TorchAudio Test (`test-torchaudio.Dockerfile`)
Tests TorchAudio's audio processing capabilities with GPU acceleration.

**Tests Include:**
- TorchAudio installation and backend verification
- Basic audio tensor operations on GPU
- Audio resampling with GPU acceleration
- Spectrogram computation
- Mel spectrogram generation
- MFCC (Mel-frequency cepstral coefficients) computation
- Batch audio processing
- Audio augmentation (time stretch, pitch shift)
- Frequency and time masking (SpecAugment)
- 30-second sustained audio processing stress test

### 3. CUDA Python Test (`test-cuda-python.Dockerfile`)
Tests low-level CUDA API functionality through cuda-python bindings.

**Tests Include:**
- CUDA driver initialization and version
- Device properties and capabilities
- CUDA context management
- Device memory allocation and transfer
- NVRTC kernel compilation and execution
- CUDA stream operations
- CUDA event operations and timing
- Memory info and usage tracking
- Custom matrix multiplication kernel
- 30-second sustained CUDA API stress test

### 4. Triton Test (`test-triton.Dockerfile`)
Tests Triton GPU programming language and JIT compiler.

**Tests Include:**
- Triton installation and configuration
- Vector addition kernel compilation
- Optimized matrix multiplication kernel
- Softmax kernel implementation
- Layer normalization kernel
- Fused attention kernel (transformer operations)
- Multiple kernel compilation and execution
- Different data types (FP32, FP16)
- Large tensor operations
- 30-second sustained kernel compilation stress test

## Building the Test Images

Build each test image separately using the following commands:

### Build PyTorch Test Image
```bash
cd /path/to/pypi/Dockerfiles
docker build -f test-torch.Dockerfile -t jetson-test-torch:latest .
```

### Build TorchAudio Test Image
```bash
docker build -f test-torchaudio.Dockerfile -t jetson-test-torchaudio:latest .
```

### Build CUDA Python Test Image
```bash
docker build -f test-cuda-python.Dockerfile -t jetson-test-cuda-python:latest .
```

### Build Triton Test Image
```bash
docker build -f test-triton.Dockerfile -t jetson-test-triton:latest .
```

**Note:** Build times can be significant (30-60 minutes per image) due to package downloads and dependencies. Ensure you have a stable internet connection.

## Running the Tests

All tests must be run with the `--runtime=nvidia` flag (or `--gpus all`) to enable GPU access.

### Run PyTorch Test
```bash
docker run --runtime=nvidia --rm jetson-test-torch:latest
```

Or with explicit GPU access:
```bash
docker run --gpus all --rm jetson-test-torch:latest
```

### Run TorchAudio Test
```bash
docker run --runtime=nvidia --rm jetson-test-torchaudio:latest
```

### Run CUDA Python Test
```bash
docker run --runtime=nvidia --rm jetson-test-cuda-python:latest
```

### Run Triton Test
```bash
docker run --runtime=nvidia --rm jetson-test-triton:latest
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

### Run All Tests Sequentially
```bash
#!/bin/bash
echo "Running all Jetson Orin package tests..."

echo "========================================="
echo "Test 1: PyTorch"
echo "========================================="
docker run --runtime=nvidia --rm jetson-test-torch:latest

echo ""
echo "========================================="
echo "Test 2: TorchAudio"
echo "========================================="
docker run --runtime=nvidia --rm jetson-test-torchaudio:latest

echo ""
echo "========================================="
echo "Test 3: CUDA Python"
echo "========================================="
docker run --runtime=nvidia --rm jetson-test-cuda-python:latest

echo ""
echo "========================================="
echo "Test 4: Triton"
echo "========================================="
docker run --runtime=nvidia --rm jetson-test-triton:latest

echo ""
echo "All tests completed!"
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

## Customizing Tests

Each Dockerfile contains the test script inline. To modify tests:

1. Extract the test script from the running container:
```bash
docker run --runtime=nvidia --rm -v $(pwd):/output jetson-test-torch:latest \
    cp /tests/test_torch_cuda.py /output/
```

2. Modify the script as needed

3. Run with custom script:
```bash
docker run --runtime=nvidia --rm -v $(pwd):/tests jetson-test-torch:latest \
    python3 /tests/test_torch_cuda.py
```

## Package Versions

The test Dockerfiles pull packages from the private PyPI index:
- Base Image: `ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0`
- PyPI Index: `https://pypi.juno-labs.com/nvcr-io-nvidia-l4t-jetpack-r36-4-0/`

Packages are built specifically for Jetson Orin with:
- Compute Capability: 8.7 (Ampere architecture)
- CUDA: 12.x (from JetPack)
- Architecture: aarch64 (ARM64)

## CI/CD Integration

These Dockerfiles can be integrated into CI/CD pipelines for automated testing:

```yaml
# Example GitHub Actions workflow
name: Test Jetson Packages
on: [push]
jobs:
  test:
    runs-on: [self-hosted, jetson-orin]
    steps:
      - uses: actions/checkout@v3
      - name: Build and test PyTorch
        run: |
          docker build -f Dockerfiles/test-torch.Dockerfile -t test-torch .
          docker run --runtime=nvidia --rm test-torch
```

## Additional Resources

- [JetPack Documentation](https://developer.nvidia.com/embedded/jetpack)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/pytorch-for-jetson/72)
- [Triton Documentation](https://triton-lang.org/)

## Support

For issues specific to:
- **Jetson hardware:** NVIDIA Developer Forums
- **Package builds:** Open an issue in this repository
- **Docker runtime:** Check nvidia-container-toolkit documentation

## License

These test Dockerfiles are provided as-is for testing purposes. Refer to individual package licenses for usage terms.
