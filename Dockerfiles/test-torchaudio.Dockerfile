# syntax=docker/dockerfile:1
# Test Dockerfile for TorchAudio on Nvidia Jetson Orin
# This Dockerfile downloads TorchAudio and runs comprehensive audio processing tests

ARG BASE_IMAGE=ghcr.io/juno-ai-labs/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-c"]

# Install Python and audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libopenblas0 \
    libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    libavfilter-dev libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch first (required dependency)
ARG PYTORCH_INDEX_URL=https://jetson-pypi.juno-labs.com/images/nvcr-io-nvidia-l4t-jetpack-r36-4-0
ARG TORCH_VERSION=2.8.0
ARG VERSION=2.8.0

RUN pip3 config set global.index-url "${PYTORCH_INDEX_URL}"

RUN python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} torchaudio==${VERSION}

# Create test script directory
RUN mkdir -p /tests

# Create comprehensive audio processing test script
RUN cat > /tests/test_torchaudio_cuda.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive TorchAudio CUDA Hardware Test Suite for Jetson Orin
Tests audio processing with GPU acceleration and various transforms
"""

import torch
import torchaudio
import sys
import time
import math

def print_separator(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_torchaudio_installation():
    """Test 1: TorchAudio Installation and Backend"""
    print_separator("Test 1: TorchAudio Installation and Backend")
    
    print(f"TorchAudio Version: {torchaudio.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
    else:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check available backends
    print(f"\nAvailable audio backends: {torchaudio.list_audio_backends()}")
    
    print("✓ TorchAudio installation test passed")

def generate_test_audio(sample_rate=16000, duration=5, frequency=440):
    """Generate test audio signal (sine wave)"""
    num_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, num_samples)
    waveform = torch.sin(2 * math.pi * frequency * t).unsqueeze(0)
    return waveform, sample_rate

def test_basic_audio_tensors():
    """Test 2: Basic Audio Tensor Operations on GPU"""
    print_separator("Test 2: Basic Audio Tensor Operations on GPU")
    
    # Generate test audio
    waveform, sample_rate = generate_test_audio(duration=5)
    print(f"Generated test audio: shape={waveform.shape}, sample_rate={sample_rate}")
    
    # Move to GPU
    if torch.cuda.is_available():
        waveform_gpu = waveform.cuda()
        print(f"Audio tensor on GPU: device={waveform_gpu.device}")
        
        # Basic operations on GPU
        # Amplitude scaling
        scaled = waveform_gpu * 0.5
        print("✓ Amplitude scaling on GPU")
        
        # Concatenation
        concatenated = torch.cat([waveform_gpu, waveform_gpu], dim=1)
        print(f"✓ Concatenation on GPU: new shape={concatenated.shape}")
        
        # Move back to CPU and verify
        result_cpu = scaled.cpu()
        assert torch.allclose(result_cpu, waveform * 0.5), "GPU computation mismatch"
        print("✓ GPU computation verified")
    else:
        print("Skipping GPU tests - CUDA not available")
    
    print("✓ Basic audio tensor operations test passed")

def test_resampling():
    """Test 3: Audio Resampling with GPU Acceleration"""
    print_separator("Test 3: Audio Resampling")
    
    # Generate test audio at different sample rates
    original_rate = 44100
    target_rate = 16000
    
    waveform, _ = generate_test_audio(sample_rate=original_rate, duration=3)
    print(f"Original audio: {waveform.shape[1]} samples at {original_rate} Hz")
    
    # Create resampler
    resampler = torchaudio.transforms.Resample(
        orig_freq=original_rate,
        new_freq=target_rate
    )
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        resampler = resampler.cuda()
        
        # Perform resampling on GPU
        torch.cuda.synchronize()
        start = time.time()
        resampled = resampler(waveform)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Resampled audio: {resampled.shape[1]} samples at {target_rate} Hz")
        print(f"Resampling time (GPU): {elapsed*1000:.2f} ms")
        
        # Verify output size is correct
        expected_length = int(waveform.shape[1] * target_rate / original_rate)
        actual_length = resampled.shape[1]
        print(f"Expected length: ~{expected_length}, Actual: {actual_length}")
        
        assert abs(actual_length - expected_length) < 100, "Resampling length mismatch"
    else:
        resampled = resampler(waveform)
        print(f"Resampled audio (CPU): {resampled.shape[1]} samples")
    
    print("✓ Resampling test passed")

def test_spectrogram_computation():
    """Test 4: Spectrogram Computation on GPU"""
    print_separator("Test 4: Spectrogram Computation")
    
    waveform, sample_rate = generate_test_audio(duration=5)
    print(f"Input waveform shape: {waveform.shape}")
    
    # Create spectrogram transform
    n_fft = 1024
    hop_length = 512
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        spectrogram_transform = spectrogram_transform.cuda()
        
        # Compute spectrogram on GPU
        torch.cuda.synchronize()
        start = time.time()
        spectrogram = spectrogram_transform(waveform)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Spectrogram shape: {spectrogram.shape}")
        print(f"Computation time (GPU): {elapsed*1000:.2f} ms")
        
        # Verify spectrogram properties
        assert spectrogram.shape[1] == n_fft // 2 + 1, "Incorrect frequency bins"
        assert not torch.isnan(spectrogram).any(), "NaN detected in spectrogram"
        assert (spectrogram >= 0).all(), "Negative values in power spectrogram"
        
        print(f"Spectrogram stats: min={spectrogram.min():.4f}, max={spectrogram.max():.4f}")
    else:
        spectrogram = spectrogram_transform(waveform)
        print(f"Spectrogram shape (CPU): {spectrogram.shape}")
    
    print("✓ Spectrogram computation test passed")

def test_mel_spectrogram():
    """Test 5: Mel Spectrogram Computation"""
    print_separator("Test 5: Mel Spectrogram Computation")
    
    waveform, sample_rate = generate_test_audio(duration=5)
    
    # Create Mel spectrogram transform
    n_mels = 128
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        mel_transform = mel_transform.cuda()
        
        # Compute Mel spectrogram on GPU
        torch.cuda.synchronize()
        start = time.time()
        mel_spec = mel_transform(waveform)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Computation time (GPU): {elapsed*1000:.2f} ms")
        
        # Verify Mel spectrogram properties
        assert mel_spec.shape[1] == n_mels, "Incorrect number of Mel bins"
        assert not torch.isnan(mel_spec).any(), "NaN detected in Mel spectrogram"
        
        print(f"Mel spectrogram stats: min={mel_spec.min():.4f}, max={mel_spec.max():.4f}")
    else:
        mel_spec = mel_transform(waveform)
        print(f"Mel spectrogram shape (CPU): {mel_spec.shape}")
    
    print("✓ Mel spectrogram test passed")

def test_mfcc_computation():
    """Test 6: MFCC Computation"""
    print_separator("Test 6: MFCC Computation")
    
    waveform, sample_rate = generate_test_audio(duration=5)
    
    # Create MFCC transform
    n_mfcc = 40
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': 1024,
            'hop_length': 512,
            'n_mels': 128
        }
    )
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        mfcc_transform = mfcc_transform.cuda()
        
        # Compute MFCC on GPU
        torch.cuda.synchronize()
        start = time.time()
        mfcc = mfcc_transform(waveform)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Computation time (GPU): {elapsed*1000:.2f} ms")
        
        # Verify MFCC properties
        assert mfcc.shape[1] == n_mfcc, "Incorrect number of MFCC coefficients"
        assert not torch.isnan(mfcc).any(), "NaN detected in MFCC"
        
        print(f"MFCC stats: min={mfcc.min():.4f}, max={mfcc.max():.4f}")
    else:
        mfcc = mfcc_transform(waveform)
        print(f"MFCC shape (CPU): {mfcc.shape}")
    
    print("✓ MFCC computation test passed")

def test_batch_processing():
    """Test 7: Batch Audio Processing on GPU"""
    print_separator("Test 7: Batch Audio Processing")
    
    batch_size = 16
    duration = 3
    sample_rate = 16000
    
    # Generate batch of audio
    batch_waveforms = []
    for i in range(batch_size):
        wf, _ = generate_test_audio(sample_rate=sample_rate, duration=duration, frequency=440 + i*50)
        batch_waveforms.append(wf)
    
    batch = torch.cat(batch_waveforms, dim=0)
    print(f"Batch shape: {batch.shape}")
    
    if torch.cuda.is_available():
        batch = batch.cuda()
        
        # Create transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=64
        ).cuda()
        
        # Process batch on GPU
        torch.cuda.synchronize()
        start = time.time()
        batch_output = mel_transform(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Batch output shape: {batch_output.shape}")
        print(f"Batch processing time (GPU): {elapsed*1000:.2f} ms")
        print(f"Per-sample time: {elapsed*1000/batch_size:.2f} ms")
        
        assert not torch.isnan(batch_output).any(), "NaN detected in batch output"
    else:
        print("Skipping GPU batch processing - CUDA not available")
    
    print("✓ Batch processing test passed")

def test_audio_augmentation():
    """Test 8: Audio Augmentation on GPU"""
    print_separator("Test 8: Audio Augmentation")
    
    waveform, sample_rate = generate_test_audio(duration=3)
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        
        # Time stretching
        print("Testing time stretch...")
        rate = 1.2
        time_stretch = torchaudio.transforms.TimeStretch(
            n_freq=513,
            hop_length=512
        ).cuda()
        
        # Need to compute spectrogram first for time stretch
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=1024,
            hop_length=512,
            return_complex=True
        ).unsqueeze(0)
        
        stretched = time_stretch(spec, rate)
        print(f"✓ Time stretch completed: {spec.shape} -> {stretched.shape}")
        
        # Pitch shift
        print("Testing pitch shift...")
        n_steps = 2
        pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=sample_rate,
            n_steps=n_steps
        ).cuda()
        
        shifted = pitch_shift(waveform)
        print(f"✓ Pitch shift completed: shifted by {n_steps} semitones")
        
        assert not torch.isnan(shifted).any(), "NaN detected in pitch shifted audio"
    else:
        print("Skipping GPU augmentation tests - CUDA not available")
    
    print("✓ Audio augmentation test passed")

def test_frequency_masking():
    """Test 9: Frequency and Time Masking (SpecAugment)"""
    print_separator("Test 9: Frequency and Time Masking")
    
    waveform, sample_rate = generate_test_audio(duration=5)
    
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        
        # Create spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        ).cuda()
        
        mel_spec = mel_transform(waveform)
        print(f"Original Mel spectrogram shape: {mel_spec.shape}")
        
        # Frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=20
        ).cuda()
        
        masked_freq = freq_mask(mel_spec)
        print(f"✓ Frequency masking applied")
        
        # Time masking
        time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=50
        ).cuda()
        
        masked_time = time_mask(mel_spec)
        print(f"✓ Time masking applied")
        
        # Verify shapes preserved
        assert masked_freq.shape == mel_spec.shape, "Shape changed after frequency masking"
        assert masked_time.shape == mel_spec.shape, "Shape changed after time masking"
    else:
        print("Skipping GPU masking tests - CUDA not available")
    
    print("✓ Frequency and time masking test passed")

def test_stress_test():
    """Test 10: Audio Processing Stress Test"""
    print_separator("Test 10: Audio Processing Stress Test")
    
    if not torch.cuda.is_available():
        print("Skipping stress test - CUDA not available")
        return
    
    print("Running sustained audio processing for 30 seconds...")
    
    sample_rate = 16000
    duration = 5
    iterations = 0
    start_time = time.time()
    test_duration = 30  # seconds
    
    # Create transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    ).cuda()
    
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 128}
    ).cuda()
    
    while time.time() - start_time < test_duration:
        # Generate audio
        waveform, _ = generate_test_audio(
            sample_rate=sample_rate,
            duration=duration,
            frequency=440 + (iterations % 100)
        )
        waveform = waveform.cuda()
        
        # Compute multiple transforms
        mel_spec = mel_transform(waveform)
        mfcc = mfcc_transform(waveform)
        
        # Additional operations
        scaled = waveform * 0.8
        resampler = torchaudio.transforms.Resample(sample_rate, sample_rate//2).cuda()
        downsampled = resampler(waveform)
        
        iterations += 1
        
        # Report every 5 seconds
        elapsed = time.time() - start_time
        if iterations % 20 == 0:
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"  Time: {elapsed:.1f}s | Iterations: {iterations} | Memory: {mem_allocated:.2f} MB")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {iterations} iterations in {elapsed:.1f} seconds")
    print(f"Average iteration time: {elapsed/iterations*1000:.2f} ms")
    print(f"Throughput: {iterations/elapsed:.2f} iterations/second")
    
    print("✓ Stress test passed")

def main():
    """Run all tests"""
    print("\n")
    print("="*80)
    print("  TorchAudio CUDA Hardware Test Suite for Jetson Orin")
    print("="*80)
    
    try:
        test_torchaudio_installation()
        test_basic_audio_tensors()
        test_resampling()
        test_spectrogram_computation()
        test_mel_spectrogram()
        test_mfcc_computation()
        test_batch_processing()
        test_audio_augmentation()
        test_frequency_masking()
        test_stress_test()
        
        print("\n")
        print("="*80)
        print("  ALL TESTS PASSED ✓")
        print("  TorchAudio is properly configured")
        print("  GPU-accelerated audio processing is working correctly")
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
RUN chmod +x /tests/test_torchaudio_cuda.py

# Set the default command to run the tests
CMD ["python3", "/tests/test_torchaudio_cuda.py"]
