# syntax=docker/dockerfile:1
# Test Dockerfile for CUDA Python on Nvidia Jetson Orin
# This Dockerfile downloads cuda-python and runs comprehensive CUDA API tests

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-c"]

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install cuda-python from the private PyPI index
ARG CUDA_PYTHON_INDEX_URL=https://jetson-pypi.juno-labs.com/images/nvcr-io-nvidia-l4t-jetpack-r36-4-0
ARG VERSION=12.6.2.post1

RUN pip3 config set global.index-url "${CUDA_PYTHON_INDEX_URL}"
RUN python3 -m pip install --no-cache-dir cuda-python==${VERSION}

# Install numpy for array operations
RUN python3 -m pip install --no-cache-dir numpy

# Create test script directory
RUN mkdir -p /tests

# Create comprehensive CUDA Python API test script
RUN cat > /tests/test_cuda_python.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive CUDA Python API Test Suite for Jetson Orin
Tests low-level CUDA API functionality and hardware interaction
"""

import sys
import time
import numpy as np
from cuda import cuda

def print_separator(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_cuda_errors(result):
    """Helper function to check CUDA errors"""
    if isinstance(result, tuple):
        err = result[0]
        if len(result) == 1:
            result = None
        else:
            result = result[1:]
    else:
        err = result
        result = None
    
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_str = cuda.cuGetErrorString(err)
        raise RuntimeError(f"CUDA Error: {error_str[1].decode() if error_str[0] == cuda.CUresult.CUDA_SUCCESS else 'Unknown'}")
    
    return result

def test_cuda_initialization():
    """Test 1: CUDA Initialization and Driver API"""
    print_separator("Test 1: CUDA Initialization and Driver API")
    
    # Initialize CUDA
    check_cuda_errors(cuda.cuInit(0))
    print("✓ CUDA initialized successfully")
    
    # Get CUDA driver version
    driver_version = check_cuda_errors(cuda.cuDriverGetVersion())
    print(f"CUDA Driver Version: {driver_version[0]}")
    
    print("✓ CUDA initialization test passed")

def test_device_properties():
    """Test 2: Device Properties and Capabilities"""
    print_separator("Test 2: Device Properties and Capabilities")
    
    # Get device count
    device_count = check_cuda_errors(cuda.cuDeviceGetCount())
    print(f"Number of CUDA devices: {device_count[0]}")
    
    if device_count[0] == 0:
        print("ERROR: No CUDA devices found!")
        sys.exit(1)
    
    # Get device handle
    device = check_cuda_errors(cuda.cuDeviceGet(0))
    print(f"\nDevice 0: {device[0]}")
    
    # Get device name
    device_name = check_cuda_errors(cuda.cuDeviceGetName(256, device[0]))
    print(f"Device Name: {device_name[0].decode()}")
    
    # Get compute capability
    major = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device[0]
    ))
    minor = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device[0]
    ))
    print(f"Compute Capability: {major[0]}.{minor[0]}")
    
    # Get memory info
    total_mem = check_cuda_errors(cuda.cuDeviceTotalMem(device[0]))
    print(f"Total Memory: {total_mem[0] / 1024**3:.2f} GB")
    
    # Get multiprocessor count
    sm_count = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        device[0]
    ))
    print(f"Multiprocessors: {sm_count[0]}")
    
    # Get max threads per block
    max_threads = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device[0]
    ))
    print(f"Max Threads per Block: {max_threads[0]}")
    
    # Get warp size
    warp_size = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        device[0]
    ))
    print(f"Warp Size: {warp_size[0]}")
    
    # Get clock rate
    clock_rate = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        device[0]
    ))
    print(f"Clock Rate: {clock_rate[0] / 1000:.2f} MHz")
    
    # Get memory clock rate
    mem_clock = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        device[0]
    ))
    print(f"Memory Clock Rate: {mem_clock[0] / 1000:.2f} MHz")
    
    # Get memory bus width
    mem_bus_width = check_cuda_errors(cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        device[0]
    ))
    print(f"Memory Bus Width: {mem_bus_width[0]} bits")
    
    print("✓ Device properties test passed")

def test_context_management():
    """Test 3: CUDA Context Management"""
    print_separator("Test 3: CUDA Context Management")
    
    # Get device
    device = check_cuda_errors(cuda.cuDeviceGet(0))
    
    # Create context
    context = check_cuda_errors(cuda.cuCtxCreate(0, device[0]))
    print(f"✓ Created CUDA context: {context[0]}")
    
    # Get current context
    current_ctx = check_cuda_errors(cuda.cuCtxGetCurrent())
    print(f"✓ Retrieved current context: {current_ctx[0]}")
    
    # Synchronize context
    check_cuda_errors(cuda.cuCtxSynchronize())
    print("✓ Context synchronized")
    
    print("✓ Context management test passed")

def test_memory_operations():
    """Test 4: Device Memory Allocation and Transfer"""
    print_separator("Test 4: Device Memory Allocation and Transfer")
    
    # Create host data
    size = 1024 * 1024  # 1M elements
    host_data = np.random.rand(size).astype(np.float32)
    print(f"Created host array: {size} elements ({host_data.nbytes / 1024**2:.2f} MB)")
    
    # Allocate device memory
    device_ptr = check_cuda_errors(cuda.cuMemAlloc(host_data.nbytes))
    print(f"✓ Allocated device memory: {host_data.nbytes / 1024**2:.2f} MB")
    
    # Copy host to device
    start = time.time()
    check_cuda_errors(cuda.cuMemcpyHtoD(device_ptr[0], host_data, host_data.nbytes))
    elapsed = time.time() - start
    print(f"✓ Host->Device transfer: {elapsed*1000:.2f} ms ({host_data.nbytes/elapsed/1024**2:.2f} MB/s)")
    
    # Allocate host memory for result
    result_data = np.zeros_like(host_data)
    
    # Copy device to host
    start = time.time()
    check_cuda_errors(cuda.cuMemcpyDtoH(result_data, device_ptr[0], host_data.nbytes))
    elapsed = time.time() - start
    print(f"✓ Device->Host transfer: {elapsed*1000:.2f} ms ({host_data.nbytes/elapsed/1024**2:.2f} MB/s)")
    
    # Verify data integrity
    if np.allclose(host_data, result_data):
        print("✓ Data integrity verified")
    else:
        raise RuntimeError("Data corruption detected!")
    
    # Free device memory
    check_cuda_errors(cuda.cuMemFree(device_ptr[0]))
    print("✓ Device memory freed")
    
    print("✓ Memory operations test passed")

def test_stream_operations():
    """Test 6: CUDA Stream Operations"""
    print_separator("Test 6: CUDA Stream Operations")
    
    # Create streams
    stream1 = check_cuda_errors(cuda.cuStreamCreate(0))
    stream2 = check_cuda_errors(cuda.cuStreamCreate(0))
    print(f"✓ Created 2 CUDA streams")
    
    # Test stream synchronization
    check_cuda_errors(cuda.cuStreamSynchronize(stream1[0]))
    check_cuda_errors(cuda.cuStreamSynchronize(stream2[0]))
    print("✓ Stream synchronization successful")
    
    # Test stream query
    result = cuda.cuStreamQuery(stream1[0])
    if result == cuda.CUresult.CUDA_SUCCESS:
        print("✓ Stream query successful - stream is idle")
    elif result == cuda.CUresult.CUDA_ERROR_NOT_READY:
        print("✓ Stream query successful - stream is busy")
    else:
        check_cuda_errors(result)
    
    # Destroy streams
    check_cuda_errors(cuda.cuStreamDestroy(stream1[0]))
    check_cuda_errors(cuda.cuStreamDestroy(stream2[0]))
    print("✓ Streams destroyed")
    
    print("✓ Stream operations test passed")

def test_event_operations():
    """Test 7: CUDA Event Operations"""
    print_separator("Test 7: CUDA Event Operations")
    
    # Create events
    start_event = check_cuda_errors(cuda.cuEventCreate(0))
    stop_event = check_cuda_errors(cuda.cuEventCreate(0))
    print("✓ Created CUDA events")
    
    # Record events
    check_cuda_errors(cuda.cuEventRecord(start_event[0], None))
    
    # Perform some work
    size = 10000000
    host_data = np.random.rand(size).astype(np.float32)
    device_ptr = check_cuda_errors(cuda.cuMemAlloc(host_data.nbytes))[0]
    check_cuda_errors(cuda.cuMemcpyHtoD(device_ptr, host_data, host_data.nbytes))
    
    check_cuda_errors(cuda.cuEventRecord(stop_event[0], None))
    check_cuda_errors(cuda.cuEventSynchronize(stop_event[0]))
    
    # Calculate elapsed time
    elapsed_ms = check_cuda_errors(cuda.cuEventElapsedTime(start_event[0], stop_event[0]))
    print(f"✓ Event timing: {elapsed_ms[0]:.2f} ms")
    
    # Cleanup
    check_cuda_errors(cuda.cuMemFree(device_ptr))
    check_cuda_errors(cuda.cuEventDestroy(start_event[0]))
    check_cuda_errors(cuda.cuEventDestroy(stop_event[0]))
    
    print("✓ Event operations test passed")

def test_memory_info():
    """Test 8: Memory Info and Usage"""
    print_separator("Test 8: Memory Info and Usage")
    
    # Get memory info
    free_mem, total_mem = check_cuda_errors(cuda.cuMemGetInfo())
    
    print(f"Total GPU Memory: {total_mem / 1024**3:.2f} GB")
    print(f"Free GPU Memory: {free_mem / 1024**3:.2f} GB")
    print(f"Used GPU Memory: {(total_mem - free_mem) / 1024**3:.2f} GB")
    print(f"Memory Usage: {((total_mem - free_mem) / total_mem * 100):.1f}%")
    
    print("✓ Memory info test passed")

def test_stress_test():
    """Test 10: CUDA API Stress Test"""
    print_separator("Test 10: CUDA API Stress Test")
    
    print("Running sustained CUDA operations for 30 seconds...")
    
    size = 1000000
    iterations = 0
    start_time = time.time()
    duration = 30  # seconds
    
    while time.time() - start_time < duration:
        # Allocate memory
        host_data = np.random.rand(size).astype(np.float32)
        device_ptr = check_cuda_errors(cuda.cuMemAlloc(host_data.nbytes))[0]
        
        # Transfer to device
        check_cuda_errors(cuda.cuMemcpyHtoD(device_ptr, host_data, host_data.nbytes))
        
        # Transfer back
        result = np.zeros_like(host_data)
        check_cuda_errors(cuda.cuMemcpyDtoH(result, device_ptr, host_data.nbytes))
        
        # Free memory
        check_cuda_errors(cuda.cuMemFree(device_ptr))
        
        iterations += 1
        
        # Report every 5 seconds
        elapsed = time.time() - start_time
        if iterations % 100 == 0:
            free_mem, total_mem = check_cuda_errors(cuda.cuMemGetInfo())
            print(f"  Time: {elapsed:.1f}s | Iterations: {iterations} | Free Memory: {free_mem/1024**2:.2f} MB")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {iterations} iterations in {elapsed:.1f} seconds")
    print(f"Average iteration time: {elapsed/iterations*1000:.2f} ms")
    print(f"Throughput: {iterations/elapsed:.2f} iterations/second")
    
    print("✓ Stress test passed")

def main():
    """Run all tests"""
    print("\n")
    print("="*80)
    print("  CUDA Python API Test Suite for Jetson Orin")
    print("="*80)
    
    try:
        test_cuda_initialization()
        test_device_properties()
        test_context_management()
        test_memory_operations()
        test_stream_operations()
        test_event_operations()
        test_memory_info()
        test_stress_test()
        
        print("\n")
        print("="*80)
        print("  ALL TESTS PASSED ✓")
        print("  CUDA Python is properly installed")
        print("  Low-level CUDA API is working correctly")
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
RUN chmod +x /tests/test_cuda_python.py

# Set the default command to run the tests
CMD ["python3", "/tests/test_cuda_python.py"]
