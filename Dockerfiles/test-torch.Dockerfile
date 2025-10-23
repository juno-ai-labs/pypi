# syntax=docker/dockerfile:1
# Test Dockerfile for PyTorch on Nvidia Jetson Orin
# This Dockerfile downloads PyTorch and runs comprehensive CUDA hardware tests

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-c"]

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch from the private PyPI index
ARG PYTORCH_INDEX_URL=https://jetson-pypi.juno-labs.com/images/nvcr-io-nvidia-l4t-jetpack-r36-4-0
RUN python3 -m pip install --no-cache-dir \
    --index-url ${PYTORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple \
    torch

# Create test script directory
RUN mkdir -p /tests

# Create comprehensive CUDA test script
RUN cat > /tests/test_torch_cuda.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive PyTorch CUDA Hardware Test Suite for Jetson Orin
Tests GPU functionality, memory operations, and compute capabilities
"""

import torch
import sys
import time
import gc

def print_separator(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_cuda_availability():
    """Test 1: CUDA Availability and Version"""
    print_separator("Test 1: CUDA Availability and Version")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    print("✓ CUDA availability test passed")

def test_basic_tensor_operations():
    """Test 2: Basic Tensor Operations on GPU"""
    print_separator("Test 2: Basic Tensor Operations on GPU")
    
    # Create tensors on GPU
    size = 1000
    x = torch.randn(size, size, device='cuda')
    y = torch.randn(size, size, device='cuda')
    
    print(f"Created {size}x{size} tensors on GPU")
    print(f"Tensor device: {x.device}")
    print(f"Tensor dtype: {x.dtype}")
    
    # Basic operations
    z = x + y
    print("✓ Addition completed")
    
    z = x * y
    print("✓ Multiplication completed")
    
    z = torch.matmul(x, y)
    print("✓ Matrix multiplication completed")
    
    # Check result validity
    assert z.device.type == 'cuda', "Result tensor not on GPU"
    assert not torch.isnan(z).any(), "NaN detected in results"
    
    print("✓ Basic tensor operations test passed")

def test_memory_management():
    """Test 3: GPU Memory Management"""
    print_separator("Test 3: GPU Memory Management")
    
    # Get initial memory stats
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated() / 1024**2
    initial_reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"Initial allocated memory: {initial_allocated:.2f} MB")
    print(f"Initial reserved memory: {initial_reserved:.2f} MB")
    
    # Allocate large tensors
    tensors = []
    for i in range(5):
        tensor = torch.randn(2048, 2048, device='cuda')
        tensors.append(tensor)
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"After allocation {i+1}: {allocated:.2f} MB allocated")
    
    # Free memory
    del tensors
    torch.cuda.empty_cache()
    gc.collect()
    
    final_allocated = torch.cuda.memory_allocated() / 1024**2
    final_reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"Final allocated memory: {final_allocated:.2f} MB")
    print(f"Final reserved memory: {final_reserved:.2f} MB")
    
    print("✓ Memory management test passed")

def test_data_transfer():
    """Test 4: CPU-GPU Data Transfer"""
    print_separator("Test 4: CPU-GPU Data Transfer")
    
    size = 5000
    
    # CPU to GPU transfer
    cpu_tensor = torch.randn(size, size)
    start = time.time()
    gpu_tensor = cpu_tensor.cuda()
    transfer_time = time.time() - start
    
    print(f"CPU -> GPU transfer time: {transfer_time*1000:.2f} ms")
    print(f"Transfer rate: {cpu_tensor.numel()*4/transfer_time/1024**2:.2f} MB/s")
    
    # GPU to CPU transfer
    start = time.time()
    cpu_result = gpu_tensor.cpu()
    transfer_time = time.time() - start
    
    print(f"GPU -> CPU transfer time: {transfer_time*1000:.2f} ms")
    print(f"Transfer rate: {gpu_tensor.numel()*4/transfer_time/1024**2:.2f} MB/s")
    
    # Verify data integrity
    assert torch.allclose(cpu_tensor, cpu_result), "Data corruption during transfer"
    
    print("✓ Data transfer test passed")

def test_matrix_operations():
    """Test 5: Intensive Matrix Operations"""
    print_separator("Test 5: Intensive Matrix Operations")
    
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Matrix multiplication benchmark
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate GFLOPS
        flops = 2 * size**3  # Matrix multiplication FLOPS
        gflops = flops / elapsed / 1e9
        
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        
        # Verify result
        assert not torch.isnan(c).any(), f"NaN detected in {size}x{size} matmul"
    
    print("\n✓ Matrix operations test passed")

def test_convolution_operations():
    """Test 6: Convolution Operations (CNN workload)"""
    print_separator("Test 6: Convolution Operations")
    
    # Simulate typical CNN layers
    batch_size = 32
    channels_in = 64
    channels_out = 128
    height, width = 224, 224
    kernel_size = 3
    
    print(f"Testing convolution: [{batch_size}, {channels_in}, {height}, {width}]")
    print(f"Kernel: {channels_out} filters of size {kernel_size}x{kernel_size}")
    
    # Create input and conv layer
    input_tensor = torch.randn(batch_size, channels_in, height, width, device='cuda')
    conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size, padding=1).cuda()
    
    # Forward pass
    torch.cuda.synchronize()
    start = time.time()
    output = conv(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Forward pass time: {elapsed*1000:.2f} ms")
    print(f"Output shape: {list(output.shape)}")
    
    # Backward pass (gradient computation)
    loss = output.sum()
    torch.cuda.synchronize()
    start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Backward pass time: {elapsed*1000:.2f} ms")
    
    assert not torch.isnan(output).any(), "NaN detected in convolution output"
    print("✓ Convolution operations test passed")

def test_mixed_precision():
    """Test 7: Mixed Precision (FP16/FP32) Operations"""
    print_separator("Test 7: Mixed Precision Operations")
    
    size = 2048
    
    # FP32 operations
    x_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)
    y_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = time.time()
    z_fp32 = torch.matmul(x_fp32, y_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    print(f"FP32 matmul time: {fp32_time*1000:.2f} ms")
    
    # FP16 operations
    x_fp16 = x_fp32.half()
    y_fp16 = y_fp32.half()
    
    torch.cuda.synchronize()
    start = time.time()
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start
    
    print(f"FP16 matmul time: {fp16_time*1000:.2f} ms")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")
    
    # Verify FP16 support
    assert z_fp16.dtype == torch.float16, "FP16 operations not working correctly"
    
    print("✓ Mixed precision test passed")

def test_multi_gpu_awareness():
    """Test 8: Multi-GPU Awareness (if available)"""
    print_separator("Test 8: Multi-GPU Awareness")
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus > 1:
        print("Testing multi-GPU tensor placement...")
        
        for i in range(num_gpus):
            tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
            print(f"✓ Created tensor on GPU {i}: {tensor.device}")
    else:
        print("Single GPU detected - multi-GPU tests skipped")
    
    print("✓ Multi-GPU awareness test passed")

def test_tensor_cores():
    """Test 9: Tensor Core Operations (if available)"""
    print_separator("Test 9: Tensor Core Operations")
    
    # Check compute capability for Tensor Core support
    props = torch.cuda.get_device_properties(0)
    compute_capability = f"{props.major}.{props.minor}"
    
    print(f"Compute Capability: {compute_capability}")
    
    # Tensor Cores available on SM 7.0+ (Volta and newer)
    # Jetson Orin has SM 8.7 (Ampere)
    if props.major >= 7:
        print("Tensor Cores supported!")
        
        # Use dimensions that are multiples of 8 for optimal Tensor Core usage
        m, n, k = 2048, 2048, 2048
        
        a = torch.randn(m, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, n, device='cuda', dtype=torch.float16)
        
        # Matrix multiplication that can use Tensor Cores
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"FP16 matmul ({m}x{k} @ {k}x{n}): {elapsed*1000:.2f} ms")
        
        flops = 2 * m * n * k
        tflops = flops / elapsed / 1e12
        print(f"Performance: {tflops:.2f} TFLOPS")
        
        assert not torch.isnan(c).any(), "NaN detected in Tensor Core operations"
    else:
        print("Tensor Cores not supported on this compute capability")
    
    print("✓ Tensor Core operations test passed")

def test_stress_test():
    """Test 10: GPU Stress Test"""
    print_separator("Test 10: GPU Stress Test")
    
    print("Running sustained load test for 30 seconds...")
    
    size = 2048
    iterations = 0
    start_time = time.time()
    duration = 30  # seconds
    
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    while time.time() - start_time < duration:
        # Intensive operations
        c = torch.matmul(a, b)
        c = c + a
        c = torch.nn.functional.relu(c)
        
        # Update tensors
        a = c
        
        iterations += 1
        
        # Report every 5 seconds
        elapsed = time.time() - start_time
        if iterations % 50 == 0:
            temp = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else "N/A"
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"  Time: {elapsed:.1f}s | Iterations: {iterations} | Memory: {mem_allocated:.2f} MB")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {iterations} iterations in {elapsed:.1f} seconds")
    print(f"Average iteration time: {elapsed/iterations*1000:.2f} ms")
    print(f"Throughput: {iterations/elapsed:.2f} iterations/second")
    
    assert not torch.isnan(c).any(), "NaN detected during stress test"
    
    print("✓ Stress test passed")

def main():
    """Run all tests"""
    print("\n")
    print("="*80)
    print("  PyTorch CUDA Hardware Test Suite for Jetson Orin")
    print("="*80)
    
    try:
        test_cuda_availability()
        test_basic_tensor_operations()
        test_memory_management()
        test_data_transfer()
        test_matrix_operations()
        test_convolution_operations()
        test_mixed_precision()
        test_multi_gpu_awareness()
        test_tensor_cores()
        test_stress_test()
        
        print("\n")
        print("="*80)
        print("  ALL TESTS PASSED ✓")
        print("  PyTorch is properly compiled with CUDA support")
        print("  Hardware acceleration is working correctly")
        print("="*80)
        print("\n")
        
        return 0
        
    except Exception as e:
        print("\n")
        print("="*80)
        print(f"  TEST FAILED ✗")
        print(f"  Error: {e}")
        print("="*80)
        print("\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make the test script executable
RUN chmod +x /tests/test_torch_cuda.py

# Set the default command to run the tests
CMD ["python3", "/tests/test_torch_cuda.py"]
