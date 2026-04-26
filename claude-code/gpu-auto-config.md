# gpu-auto-config

## 概述

自动检测运行环境的 GPU 加速能力（CUDA、MPS、ROCm），并自动配置 video-analyzer 各组件（Whisper 转录、LLM 推理、帧处理）使用最优计算设置。

## 适用场景

- 在新机器上首次部署 video-analyzer
- 处理速度异常缓慢（可能在 CPU 上运行）
- 本地与云端环境之间迁移
- 用户不确定硬件能力

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| force_device | string | 自动检测 | 强制覆盖："cuda" / "mps" / "cpu" |
| whisper_compute_type | string | 自动检测 | 强制精度："float16" / "int8" / "float32" |
| verbose | bool | true | 是否打印检测到的硬件详情 |

## 核心指令

实现硬件自动检测，按以下优先级选择设备：CUDA > MPS（Apple Silicon）> ROCm > CPU。

### 1. 设备检测逻辑

```python
import torch
import subprocess
import os

def detect_optimal_device() -> tuple[str, str, str]:
    """返回 (device, compute_type, llm_backend)"""
    # CUDA
    if torch.cuda.is_available():
        device = "cuda"
        capability = torch.cuda.get_device_capability()
        compute_type = "float16" if capability[0] >= 7 else "int8"
        return device, compute_type, "cuda"

    # Apple Silicon MPS
    if torch.backends.mps.is_available():
        return "mps", "float16", "mps"

    # AMD ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "cuda", "float16", "rocm"

    # CPU
    device = "cpu"
    try:
        import cpuinfo
        flags = cpuinfo.get_cpu_info().get('flags', [])
        compute_type = "int8" if 'avx2' in flags else "float32"
    except ImportError:
        compute_type = "float32"
    return device, compute_type, "cpu"
```

### 2. Ollama GPU 检测

```python
def detect_ollama_gpu() -> bool:
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "%/GPU" in result.stdout or any("GPU" in line for line in result.stdout.split("\n"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False
```

### 3. 最优并发数

```python
def get_optimal_workers(device: str) -> int:
    if device == "cuda":
        return torch.cuda.device_count() * 2
    elif device == "mps":
        return 2
    return max(1, os.cpu_count() // 2)
```

### 4. 配置集成

更新 `video_analyzer/config.py`：

```python
def apply_hardware_defaults(config: dict) -> dict:
    device, compute_type, _ = detect_optimal_device()
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["device"] = device
    config["audio"]["compute_type"] = compute_type
    if detect_ollama_gpu():
        config.setdefault("clients", {})["ollama_gpu"] = True
    config["max_workers"] = get_optimal_workers(device)
    return config
```

### 5. 音频处理器更新

```python
class AudioProcessor:
    def __init__(self, model_size="base", device=None, compute_type=None):
        from .config import detect_optimal_device
        if device is None or compute_type is None:
            d, c, _ = detect_optimal_device()
            device = device or d
            compute_type = compute_type or c
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
```

### 6. 启动信息输出

```python
def print_hardware_info():
    device, compute_type, backend = detect_optimal_device()
    print(f"硬件检测:")
    print(f"  设备: {device}")
    print(f"  Whisper compute_type: {compute_type}")
    print(f"  并行 workers: {get_optimal_workers(device)}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == "mps":
        print(f"  后端: Apple Metal Performance Shaders")
    print(f"  Ollama GPU: {'是' if detect_ollama_gpu() else '否/未知'}")
```

## 实现要点

- 可选依赖：`py-cpuinfo`（用于 CPU 特性检测）
- 支持 `force_device` 参数强制覆盖自动检测结果
- 配置保存到 `config.json` 供后续运行复用

## 验证清单

- [ ] CUDA 机器上 Whisper 使用 GPU，`torch.cuda.is_available()` 为 True
- [ ] M1/M2/M3 Mac 上 Whisper 使用 MPS，转录速度明显快于 CPU
- [ ] 纯 CPU 机器优雅降级到 int8/float32
- [ ] Ollama GPU 状态正确报告
- [ ] 配置保存后下次运行自动复用

## 示例用法

```
/gpu-auto-config
/gpu-auto-config force_device=cuda whisper_compute_type=float16
/gpu-auto-config verbose=true
```
