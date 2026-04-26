# adaptive-keyframe-extraction

## 概述

将 video-analyzer 中基于固定阈值的朴素帧差分替换为自适应、流式感知的关键帧提取器。可应对暗光场景、快速剪辑、慢速平移等多样化视频内容，同时避免长视频 OOM。

## 适用场景

- 关键帧遗漏重要场景变化
- 关键帧中充斥相似帧
- 处理长视频时内存不足
- 视频内容运动速度变化大

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| strategy | string | "adaptive-threshold" | 提取策略："adaptive-threshold" / "histogram" / "multi-frame" |
| max_memory_mb | float | 512 | 帧缓冲内存预算（MB） |
| min_scene_duration_sec | float | 1.0 | 相邻关键帧最短时间间隔（秒） |
| sensitivity | float | 0.5 | 检测灵敏度（0-1） |

## 核心指令

替换 `video_analyzer/frame.py` 中的 `VideoProcessor`，解决以下问题：

1. `frame_candidates` 存储完整 `np.ndarray` — 长视频 OOM
2. `FRAME_DIFFERENCE_THRESHOLD = 10.0` 固定值 — 暗光/强光视频失效
3. 仅与前一帧比较 — 漏检渐变场景过渡
4. 两遍逻辑（采样后排序）效率低

### 1. 内存高效的流式提取器

用元数据替代完整帧存储，使用 `heapq` 实现单遍 top-K：

```python
import heapq
from dataclasses import dataclass

@dataclass
class FrameCandidate:
    frame_number: int
    timestamp: float
    score: float

class StreamingKeyframeExtractor:
    def __init__(self, max_frames=10, sensitivity=0.5, min_scene_duration_sec=1.0):
        self.max_frames = max_frames
        self.sensitivity = sensitivity
        self.min_scene_duration = min_scene_duration_sec
        self._candidates = []  # 最小堆 (score, frame_number, timestamp)
        self._recent_scores = []

    def process_frame(self, frame_number: int, timestamp: float, score: float) -> bool:
        if len(self._recent_scores) > 100:
            self._recent_scores.pop(0)
        self._recent_scores.append(score)

        threshold = (np.percentile(self._recent_scores, 75) * self.sensitivity
                     if len(self._recent_scores) >= 10 else 10.0)

        if score > threshold:
            heapq.heappush(self._candidates, (score, frame_number, timestamp))
            if len(self._candidates) > self.max_frames * 2:
                heapq.heappop(self._candidates)
            return True
        return False

    def get_keyframes(self) -> list[FrameCandidate]:
        sorted_candidates = sorted(self._candidates, key=lambda x: x[1])
        filtered = []
        last_ts = -self.min_scene_duration
        for score, fn, ts in sorted_candidates:
            if ts - last_ts >= self.min_scene_duration:
                filtered.append(FrameCandidate(fn, ts, score))
                last_ts = ts
        return filtered[:self.max_frames]
```

### 2. 自适应阈值

```python
def _compute_adaptive_threshold(self, scores: list[float], sensitivity: float = 0.5) -> float:
    if len(scores) < 10:
        return 10.0
    median = np.median(scores)
    std = np.std(scores)
    return max(5.0, median + sensitivity * std)
```

### 3. 直方图差异（光照不变）

```python
def _compute_histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
    h1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
```

### 4. 多帧差异（渐变检测）

```python
def _compute_multi_frame_diff(self, frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    return self._compute_difference(avg_frame, frames[-1])
```

## 实现要点

- 创建 `OptimizedVideoProcessor` 类或替换现有 `VideoProcessor`
- 在 `default_config.json` 中添加配置节：

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

## 验证清单

- [ ] 2 小时视频处理不触发 OOM
- [ ] 暗光视频生成有意义的关键帧
- [ ] 快速动作视频捕获关键时刻
- [ ] 慢速平移视频避免冗余帧
- [ ] 输出帧数不超过 `max_frames` 限制

## 示例用法

```
/adaptive-keyframe-extraction strategy=adaptive-threshold sensitivity=0.6
/adaptive-keyframe-extraction strategy=histogram max_memory_mb=256
/adaptive-keyframe-extraction strategy=multi-frame min_scene_duration_sec=2.0
```
