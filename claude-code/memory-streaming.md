# memory-streaming

## 概述

为 video-analyzer 实现分块、流式处理，支持长视频（1 小时以上）在不耗尽内存的情况下完成分析。包含分块视频处理、流式音频转录、流式帧分析和内存监控。

## 适用场景

- 处理超过 30 分钟的长视频
- 遇到 `MemoryError` 或 OOM 崩溃
- 在内存有限（< 8GB）的机器上运行
- 连续处理多个视频

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| chunk_duration_sec | float | 300 | 每个处理分块的时长（秒） |
| max_ram_mb | float | 1024 | 内存预算上限（MB） |
| cleanup_interval | int | 100 | 两次垃圾回收提示之间的帧数 |

## 核心指令

将视频分析流水线改造为分块、流式、内存受限的实现。

### 1. 分块视频处理器

```python
import gc
from pathlib import Path
from typing import Iterator
import cv2

class ChunkedVideoProcessor:
    def __init__(self, video_path: str, output_dir: str,
                 chunk_duration_sec: float = 300.0, max_ram_mb: float = 1024.0):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.chunk_duration_sec = chunk_duration_sec
        self.max_ram_mb = max_ram_mb
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def stream_chunks(self) -> Iterator[dict]:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        chunk_size_frames = int(self.chunk_duration_sec * fps)
        chunk_idx = 0

        while True:
            frames = []
            frame_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            for _ in range(chunk_size_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if not frames:
                break

            frame_end = frame_start + len(frames)
            yield {
                "chunk_idx": chunk_idx,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "timestamp_start": frame_start / fps,
                "timestamp_end": frame_end / fps,
                "frames": frames,
                "fps": fps,
            }
            del frames
            if chunk_idx % 2 == 0:
                gc.collect()
            chunk_idx += 1
        cap.release()
```

### 2. 流式关键帧提取（按分块）

```python
    def extract_keyframes_streaming(self, target_frames_total: int = 10) -> Iterator[dict]:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        num_chunks = max(1, int(duration_sec / self.chunk_duration_sec))
        frames_per_chunk = max(1, target_frames_total // num_chunks)
        chunk_idx = 0

        while True:
            chunk_dir = self.output_dir / f"chunk_{chunk_idx:04d}"
            chunk_dir.mkdir(exist_ok=True)
            extractor = StreamingKeyframeExtractor(max_frames=frames_per_chunk, sensitivity=0.5)
            prev_frame = None
            frame_in_chunk = 0

            for _ in range(int(self.chunk_duration_sec * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                if prev_frame is not None:
                    score = extractor._compute_difference(prev_frame, frame)
                    timestamp = (chunk_idx * self.chunk_duration_sec) + (frame_in_chunk / fps)
                    extractor.process_frame(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, timestamp, score)
                prev_frame = frame
                frame_in_chunk += 1

            candidates = extractor.get_keyframes()
            for candidate in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
                ret, frame = cap.read()
                if ret:
                    output_path = chunk_dir / f"frame_{candidate.frame_number:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    yield {"path": str(output_path), "timestamp": candidate.timestamp, "chunk_idx": chunk_idx}

            chunk_idx += 1
            del extractor, prev_frame
            gc.collect()
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= total_frames:
                break
        cap.release()
```

### 3. 流式音频转录

```python
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

class StreamingAudioProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="float32",
                 chunk_length_sec: float = 30.0):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.chunk_length_sec = chunk_length_sec

    def transcribe_streaming(self, audio_path: str) -> Iterator[dict]:
        info = sf.info(audio_path)
        total_duration = info.duration
        sr = info.samplerate
        chunk_samples = int(self.chunk_length_sec * sr)
        offset = 0.0

        while offset < total_duration:
            data, _ = sf.read(audio_path, start=int(offset * sr), frames=chunk_samples, dtype="float32")
            if len(data) == 0:
                break
            if len(data) < chunk_samples:
                data = np.pad(data, (0, chunk_samples - len(data)))

            segments, _ = self.model.transcribe(data, language="en", vad_filter=True, word_timestamps=True)
            for segment in segments:
                yield {
                    "text": segment.text,
                    "start": segment.start + offset,
                    "end": segment.end + offset,
                    "confidence": segment.avg_logprob,
                }
            offset += self.chunk_length_sec
            del data, segments
            gc.collect()
```

### 4. 内存监控

```python
import psutil
import os

class MemoryMonitor:
    def __init__(self, max_ram_mb: float, check_interval: int = 10):
        self.max_ram_mb = max_ram_mb
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self._call_count = 0

    def check(self, context: str = ""):
        self._call_count += 1
        if self._call_count % self.check_interval != 0:
            return
        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        if mem_mb > self.max_ram_mb:
            gc.collect()
            mem_mb_after = self.process.memory_info().rss / (1024 * 1024)
            if mem_mb_after > self.max_ram_mb:
                raise MemoryError(f"内存超限: {mem_mb_after:.0f}MB > {self.max_ram_mb}MB ({context})")
```

## 实现要点

- CLI 新增参数：`--chunk-duration`、`--max-ram-mb`、`--stream-output`
- 流式分析结果使用 JSONL 格式追加写入，而非单个大 JSON
- 分块结果可在最后合并为完整输出

## 验证清单

- [ ] 2 小时视频在 8GB 内存机器上处理不触发 OOM
- [ ] 内存使用始终低于 `max_ram_mb` 阈值
- [ ] 分块结果可合并为完整输出
- [ ] 流式输出文件为有效 JSONL
- [ ] 支持从中间状态恢复处理

## 示例用法

```
/memory-streaming chunk_duration_sec=300 max_ram_mb=1024
/memory-streaming chunk_duration_sec=600 max_ram_mb=512 cleanup_interval=50
```
