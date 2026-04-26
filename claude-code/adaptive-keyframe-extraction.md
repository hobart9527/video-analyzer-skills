# adaptive-keyframe-extraction

Improve video keyframe extraction with adaptive thresholds, memory-efficient streaming, and scene-change detection.

## Description

Replaces the naive fixed-threshold frame differencing in video-analyzer with an adaptive, streaming-aware keyframe extractor that handles diverse video content (dark scenes, fast cuts, slow pans) without OOM errors.

## When to use

- When keyframes are missing important scene changes
- When keyframes are dominated by similar-looking frames
- When processing long videos causes memory issues
- When video content has varying motion speeds

## Parameters

- strategy: "adaptive-threshold" | "histogram" | "optical-flow" | "scene-detect"
- max_memory_mb: Memory budget for frame buffering (default: 512)
- min_scene_duration_sec: Minimum time between keyframes (default: 1.0)
- sensitivity: Detection sensitivity 0-1 (default: 0.5)

## Current Problems

The existing `VideoProcessor` in `video_analyzer/frame.py` has these issues:
1. `frame_candidates` stores FULL `np.ndarray` objects — OOM on long videos
2. `FRAME_DIFFERENCE_THRESHOLD = 10.0` is fixed — fails on dark/bright videos
3. Only compares to previous frame — misses gradual scene transitions
4. Two-pass logic (sample then sort) is inefficient

## Implementation

### 1. Memory-Efficient Streaming Extractor

Replace full-frame storage with metadata-only tracking:

```python
import heapq
from dataclasses import dataclass
from typing import Iterator
import cv2
import numpy as np

@dataclass
class FrameCandidate:
    frame_number: int
    timestamp: float
    score: float
    # NO image data stored here!

class StreamingKeyframeExtractor:
    def __init__(self, max_frames=10, max_memory_mb=512, min_scene_duration_sec=1.0):
        self.max_frames = max_frames
        self.max_memory_mb = max_memory_mb
        self.min_scene_duration = min_scene_duration_sec
        self._candidates = []  # min-heap of (score, frame_number, timestamp)

    def process_frame(self, frame_number: int, timestamp: float,
                      prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """Process a single frame. Returns True if it's a keyframe candidate."""
        score = self._compute_difference(prev_frame, curr_frame)

        # Adaptive threshold: use percentile of recent scores
        if hasattr(self, '_recent_scores'):
            self._recent_scores.append(score)
            if len(self._recent_scores) > 100:
                self._recent_scores.pop(0)
            threshold = np.percentile(self._recent_scores, 75) * self.sensitivity
        else:
            self._recent_scores = [score]
            threshold = 10.0  # fallback

        if score > threshold:
            heapq.heappush(self._candidates, (score, frame_number, timestamp))
            # Evict lowest score if over limit
            if len(self._candidates) > self.max_frames * 2:
                heapq.heappop(self._candidates)
            return True
        return False

    def get_keyframes(self, cap) -> list[FrameCandidate]:
        """Extract top frames sorted by time."""
        # Sort by frame number (time order)
        sorted_candidates = sorted(self._candidates, key=lambda x: x[1])
        # Apply min_scene_duration filter
        filtered = []
        last_ts = -self.min_scene_duration
        for score, fn, ts in sorted_candidates:
            if ts - last_ts >= self.min_scene_duration:
                filtered.append(FrameCandidate(fn, ts, score))
                last_ts = ts
        # Limit to max_frames
        return filtered[:self.max_frames]
```

### 2. Adaptive Threshold Based on Scene Statistics

```python
def _compute_adaptive_threshold(self, scores: list[float], sensitivity: float = 0.5) -> float:
    """Compute adaptive threshold using scene statistics."""
    if len(scores) < 10:
        return 10.0  # default for short videos

    median = np.median(scores)
    std = np.std(scores)
    # Threshold = median + sensitivity * std
    # This adapts to video's natural motion level
    return max(5.0, median + sensitivity * std)
```

### 3. Histogram-Based Difference (Handles lighting changes better)

```python
def _compute_histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compare color histograms instead of raw pixels."""
    h1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
```

### 4. Multi-Frame Difference (Detects gradual transitions)

```python
def _compute_multi_frame_diff(self, frames: list[np.ndarray]) -> float:
    """Compare current frame against average of last N frames."""
    if len(frames) < 2:
        return 0.0
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    return self._compute_difference(avg_frame, frames[-1])
```

## Integration into video_analyzer/frame.py

Replace the existing `VideoProcessor` class or create an optimized subclass:

```python
class OptimizedVideoProcessor:
    def __init__(self, video_path: str, output_dir: str, model=None,
                 max_frames: int = 10, strategy: str = "adaptive-threshold",
                 sensitivity: float = 0.5):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_frames = max_frames
        self.strategy = strategy
        self.sensitivity = sensitivity

    def extract_keyframes(self, target_frames: int = None) -> list[Frame]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(target_frames or self.max_frames, self.max_frames)

        extractor = StreamingKeyframeExtractor(
            max_frames=max_frames,
            sensitivity=self.sensitivity
        )

        prev_frame = None
        frame_buffer = []  # For multi-frame diff
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                if self.strategy == "histogram":
                    score = extractor._compute_histogram_diff(prev_frame, frame)
                elif self.strategy == "multi-frame" and len(frame_buffer) >= 3:
                    score = extractor._compute_multi_frame_diff(frame_buffer[-3:] + [frame])
                else:
                    score = extractor._compute_difference(prev_frame, frame)

                extractor.process_frame(frame_number, frame_number / fps, prev_frame, frame)

            prev_frame = frame
            frame_buffer.append(frame)
            if len(frame_buffer) > 5:
                frame_buffer.pop(0)

            frame_number += 1

        cap.release()

        # Extract and save only the selected keyframes
        candidates = extractor.get_keyframes(cap)
        keyframes = []

        cap = cv2.VideoCapture(self.video_path)
        for candidate in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
            ret, frame = cap.read()
            if ret:
                output_path = self.output_dir / f"frame_{candidate.frame_number:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                keyframes.append(Frame(
                    frame_number=candidate.frame_number,
                    path=str(output_path),
                    timestamp=candidate.timestamp,
                    difference_score=candidate.score
                ))
        cap.release()

        return keyframes
```

## Configuration Update

Add to `default_config.json`:
```json
{
  "frame_extraction": {
    "strategy": "adaptive-threshold",
    "sensitivity": 0.5,
    "min_scene_duration_sec": 1.0,
    "max_memory_mb": 512
  }
}
```

## Verification

- [ ] Process a 2-hour video without OOM
- [ ] Dark scene video produces meaningful keyframes
- [ ] Fast-action video captures key moments
- [ ] Slow-pan video avoids redundant frames
- [ ] Output frame count matches `max_frames` limit
