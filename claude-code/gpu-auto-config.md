# gpu-auto-config

Auto-detect available hardware (CUDA, MPS, CPU) and optimize video-analyzer settings accordingly.

## Description

Analyzes the runtime environment for GPU acceleration capabilities and automatically configures video-analyzer components (Whisper transcription, LLM inference, frame processing) to use optimal compute settings.

## When to use

- When setting up video-analyzer on a new machine
- When performance is unexpectedly slow (might be running on CPU)
- When migrating between local and cloud environments
- When the user is unsure about their hardware capabilities

## Parameters

- force_device: Override auto-detection ("cuda" | "mps" | "cpu")
- whisper_compute_type: Override Whisper precision ("float16" | "int8" | "float32")
- verbose: Print detected hardware details (default: true)

## Detection Logic

```python
import torch
import subprocess
import os

def detect_optimal_device() -> tuple[str, str]:
    """
    Returns (device, compute_type, llm_backend)
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    # Check CUDA
    if torch.cuda.is_available():
        device = "cuda"
        # Check if float16 is supported
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Turing and newer
            compute_type = "float16"
        else:
            compute_type = "int8"
        llm_backend = "cuda"
        return device, compute_type, llm_backend

    # Check Apple Silicon MPS
    if torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float16"
        llm_backend = "mps"
        return device, compute_type, llm_backend

    # Check for AMD ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        device = "cuda"  # ROCm uses cuda device string
        compute_type = "float16"
        llm_backend = "rocm"
        return device, compute_type, llm_backend

    # CPU fallback
    device = "cpu"
    # Check for AVX2 support for faster CPU inference
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        if 'avx2' in flags:
            compute_type = "int8"  # int8 is faster on CPU with AVX2
        else:
            compute_type = "float32"
    except ImportError:
        compute_type = "float32"

    llm_backend = "cpu"
    return device, compute_type, llm_backend


def detect_ollama_gpu() -> bool:
    """Check if Ollama is using GPU acceleration."""
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            # Ollama ps shows GPU memory if using GPU
            return "%/GPU" in output or any("GPU" in line for line in output.split("\n"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def get_optimal_workers(device: str) -> int:
    """Get optimal number of parallel workers based on device."""
    if device == "cuda":
        # Use number of GPUs * 2 for I/O overlap
        return torch.cuda.device_count() * 2
    elif device == "mps":
        # MPS benefits from limited concurrency
        return 2
    else:
        # CPU: use half of cores for Whisper, leave room for other tasks
        return max(1, os.cpu_count() // 2)
```

## Configuration Updates

### Update `video_analyzer/config.py`:

```python
def apply_hardware_defaults(config: dict) -> dict:
    """Auto-populate hardware-optimized defaults."""
    device, compute_type, llm_backend = detect_optimal_device()

    # Whisper settings
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["device"] = device
    config["audio"]["compute_type"] = compute_type

    # LLM settings
    if "clients" not in config:
        config["clients"] = {}

    # Check if Ollama is GPU-accelerated
    if detect_ollama_gpu():
        config["clients"]["ollama_gpu"] = True

    # Parallelism settings
    config["max_workers"] = get_optimal_workers(device)

    return config
```

### Update `video_analyzer/audio_processor.py`:

```python
class AudioProcessor:
    def __init__(self, model_size="base", device=None, compute_type=None):
        from .config import detect_optimal_device

        if device is None or compute_type is None:
            detected_device, detected_compute, _ = detect_optimal_device()
            device = device or detected_device
            compute_type = compute_type or detected_compute

        self.device = device
        self.compute_type = compute_type

        # faster_whisper auto-detects but we explicitly set for clarity
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
```

### Update CLI output:

```python
def print_hardware_info():
    device, compute_type, backend = detect_optimal_device()
    print(f"Hardware Detection:")
    print(f"  Device: {device}")
    print(f"  Whisper compute_type: {compute_type}")
    print(f"  Parallel workers: {get_optimal_workers(device)}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == "mps":
        print(f"  Backend: Apple Metal Performance Shaders")
    print(f"  Ollama GPU: {'Yes' if detect_ollama_gpu() else 'No/Unknown'}")
```

## Ollama GPU Configuration Helper

```python
def configure_ollama_gpu():
    """Provide instructions for enabling GPU in Ollama."""
    tips = []

    if sys.platform == "darwin":
        tips.append("macOS: Ollama automatically uses Metal on Apple Silicon")
    elif sys.platform == "linux":
        tips.append("Linux: Ensure nvidia-docker or CUDA drivers are installed")
        tips.append("Set OLLAMA_GPU_OVERHEAD env var if VRAM is limited")
    elif sys.platform == "win32":
        tips.append("Windows: Use WSL2 with CUDA support for GPU acceleration")

    # Check OLLAMA_HOST
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    if ollama_host != "localhost":
        tips.append(f"Remote Ollama at {ollama_host}: GPU status depends on remote server")

    return "\n".join(tips)
```

## Verification

- [ ] On CUDA machine: Whisper uses GPU, `torch.cuda.is_available()` is True
- [ ] On M1/M2/M3 Mac: Whisper uses MPS, transcription is faster than CPU
- [ ] On CPU-only machine: Falls back gracefully to int8/float32
- [ ] Ollama GPU status is correctly reported
- [ ] Configuration is saved for subsequent runs
