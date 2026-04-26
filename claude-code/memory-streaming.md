# memory-streaming

Add streaming and memory-efficient processing to handle long videos without OOM errors.

## Description

Implements chunked processing, generator-based pipelines, and memory-bounded operations throughout video-analyzer to support long videos (1+ hours) without exhausting RAM.

## When to use

- When processing videos longer than 30 minutes
- When encountering `MemoryError` or OOM crashes
- When running on machines with limited RAM (<8GB)
- When processing multiple videos in sequence

## Parameters

- chunk_duration_sec: Duration of each processing chunk (default: 300 = 5 min)
- max_ram_mb: Maximum RAM budget (default: 1024)
- cleanup_interval: Frames processed between garbage collection hints (default: 100)

## Implementation

### 1. Chunked Video Processing

Instead of processing the entire video at once, split into time chunks:

```python
import gc
from pathlib import Path
from typing import Iterator, List
import cv2
import numpy as np

class ChunkedVideoProcessor:
    def __init__(self, video_path: str, output_dir: str,
                 chunk_duration_sec: float = 300.0,
                 max_ram_mb: float = 1024.0):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.chunk_duration_sec = chunk_duration_sec
        self.max_ram_mb = max_ram_mb
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def stream_chunks(self) -> Iterator[dict]:
        """Yield video chunks with metadata. Each chunk is processed independently."""
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
            timestamp_start = frame_start / fps
            timestamp_end = frame_end / fps

            yield {
                "chunk_idx": chunk_idx,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "frames": frames,
                "fps": fps,
            }

            # Explicit cleanup
            del frames
            if chunk_idx % 2 == 0:
                gc.collect()

            chunk_idx += 1

        cap.release()
```

### 2. Streaming Keyframe Extraction (Per-Chunk)

```python
    def extract_keyframes_streaming(self, target_frames_total: int = 10) -> Iterator[dict]:
        """Stream keyframes without loading entire video."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        # Distribute target frames across chunks
        num_chunks = max(1, int(duration_sec / self.chunk_duration_sec))
        frames_per_chunk = max(1, target_frames_total // num_chunks)

        chunk_idx = 0
        while True:
            chunk_dir = self.output_dir / f"chunk_{chunk_idx:04d}"
            chunk_dir.mkdir(exist_ok=True)

            # Extract keyframes for this chunk only
            extractor = StreamingKeyframeExtractor(
                max_frames=frames_per_chunk,
                sensitivity=0.5
            )

            prev_frame = None
            frame_in_chunk = 0

            for _ in range(int(self.chunk_duration_sec * fps)):
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    score = extractor._compute_difference(prev_frame, frame)
                    timestamp = (chunk_idx * self.chunk_duration_sec) + (frame_in_chunk / fps)
                    extractor.process_frame(
                        frame_number=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                        timestamp=timestamp,
                        prev_frame=prev_frame,
                        curr_frame=frame
                    )

                prev_frame = frame
                frame_in_chunk += 1

            # Save only this chunk's keyframes
            candidates = extractor.get_keyframes(cap)
            for candidate in candidates:
                # Re-read frame at position
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
                ret, frame = cap.read()
                if ret:
                    output_path = chunk_dir / f"frame_{candidate.frame_number:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    yield {
                        "path": str(output_path),
                        "timestamp": candidate.timestamp,
                        "chunk_idx": chunk_idx,
                        **candidate.__dict__ if hasattr(candidate, '__dict__') else {}
                    }

            # Move to next chunk start (already at end from reads)
            chunk_idx += 1
            del extractor, prev_frame
            gc.collect()

            # Check if we've read all frames
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= total_frames:
                break

        cap.release()
```

### 3. Streaming Audio Transcription (Chunked)

```python
from faster_whisper import WhisperModel
import numpy as np

class StreamingAudioProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="float32",
                 chunk_length_sec: float = 30.0):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.chunk_length_sec = chunk_length_sec

    def transcribe_streaming(self, audio_path: str) -> Iterator[dict]:
        """Transcribe audio in chunks, yielding segments as they're ready."""
        import soundfile as sf

        info = sf.info(audio_path)
        total_duration = info.duration
        sr = info.samplerate

        chunk_samples = int(self.chunk_length_sec * sr)
        offset = 0.0

        while offset < total_duration:
            # Read chunk
            data, _ = sf.read(
                audio_path,
                start=int(offset * sr),
                frames=chunk_samples,
                dtype="float32"
            )

            if len(data) == 0:
                break

            # Pad last chunk if needed
            if len(data) < chunk_samples:
                data = np.pad(data, (0, chunk_samples - len(data)))

            # Transcribe chunk
            segments, info = self.model.transcribe(
                data,
                language="en",
                vad_filter=True,
                word_timestamps=True
            )

            for segment in segments:
                yield {
                    "text": segment.text,
                    "start": segment.start + offset,
                    "end": segment.end + offset,
                    "words": [
                        {"word": w.word, "start": w.start + offset, "end": w.end + offset}
                        for w in (segment.words or [])
                    ],
                    "confidence": segment.avg_logprob,
                }

            offset += self.chunk_length_sec
            del data, segments
            gc.collect()
```

### 4. Streaming Frame Analysis Results

Instead of accumulating all analyses in memory, stream results to disk:

```python
import json

class StreamingVideoAnalyzer:
    def __init__(self, analyzer, output_file: str):
        self.analyzer = analyzer
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._analyses = []
        self._flush_every = 5

    async def analyze_frame_streaming(self, frame, previous_analyses=None) -> str:
        """Analyze frame and periodically flush to disk."""
        result = await self.analyzer.analyze_frame_async(frame, previous_analyses)
        self._analyses.append({
            "frame_number": getattr(frame, "frame_number", 0),
            "timestamp": getattr(frame, "timestamp", 0),
            "analysis": result
        })

        if len(self._analyses) >= self._flush_every:
            await self._flush()

        return result

    async def _flush(self):
        """Append accumulated analyses to disk."""
        mode = "a" if self.output_file.exists() else "w"
        with open(self.output_file, mode) as f:
            for analysis in self._analyses:
                f.write(json.dumps(analysis) + "\n")
        self._analyses.clear()

    async def close(self):
        if self._analyses:
            await self._flush()
```

### 5. Memory Monitor

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
            import gc
            gc.collect()
            mem_mb_after = self.process.memory_info().rss / (1024 * 1024)
            if mem_mb_after > self.max_ram_mb:
                raise MemoryError(
                    f"Memory limit exceeded: {mem_mb_after:.0f}MB > {self.max_ram_mb}MB "
                    f"({context}). Consider reducing chunk_duration_sec or max_frames."
                )
```

## CLI Integration

Add streaming options to CLI:
```python
parser.add_argument("--chunk-duration", type=float, default=300.0,
                    help="Process video in chunks of N seconds")
parser.add_argument("--max-ram-mb", type=float, default=1024.0,
                    help="Maximum RAM usage in MB")
parser.add_argument("--stream-output", action="store_true",
                    help="Stream results to output file instead of buffering")
```

## Verification

- [ ] 2-hour video processes without OOM on 8GB RAM machine
- [ ] Memory usage stays below `max_ram_mb` threshold
- [ ] Chunked results can be recombined into full output
- [ ] Streaming output file is valid JSONL
- [ ] Processing can resume from intermediate state
