# parallel-frame-analysis

Convert sequential frame-by-frame video analysis to parallel/batch processing for significant speedup.

## Description

Optimizes the video-analyzer's frame analysis stage by replacing the sequential `for frame in frames: analyzer.analyze_frame(frame)` loop with concurrent processing. Frames that don't depend on each other can be analyzed in parallel; context-dependent frames use a sliding window approach.

## When to use

- When analyzing videos with many keyframes (>20)
- When using cloud LLM APIs that support higher concurrency
- When local GPU has spare capacity for parallel inference
- When processing batch video jobs

## Parameters

- concurrency: Maximum parallel requests (default: 4, max: 16)
- mode: "parallel-independent" | "sliding-window" | "batch-api"
- window_size: Number of previous frames to include as context (default: 3)
- provider: "ollama" | "openai" | "auto" (determines batch capabilities)

## Instructions

1. Locate the frame analysis loop in `video_analyzer/cli.py` and `video_analyzer/analyzer.py`
2. Determine the dependency model: does the LLM need ALL previous frames or just recent context?
3. Implement the selected `mode`:

### Mode: parallel-independent

For frames that can be analyzed independently (no cross-frame context):

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def analyze_frames_parallel(analyzer, frames, max_concurrency=4):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(frame):
        async with semaphore:
            return await analyzer.analyze_frame_async(frame)

    tasks = [analyze_one(f) for f in frames]
    return await asyncio.gather(*tasks)
```

### Mode: sliding-window

For context-dependent analysis, limit history to prevent prompt explosion:

```python
async def analyze_frames_sliding_window(analyzer, frames, window_size=3):
    results = []
    for i, frame in enumerate(frames):
        # Only include last N frames as context
        recent_context = results[-window_size:] if i > 0 else []
        result = await analyzer.analyze_frame_async(
            frame,
            previous_analyses=recent_context  # pass limited context
        )
        results.append(result)
    return results
```

### Mode: batch-api

For OpenAI-compatible APIs that support batch requests:

```python
async def analyze_frames_batch(client, frames, batch_size=8):
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        messages = [build_frame_message(f) for f in batch]
        batch_results = await client.batch_generate(messages)
        results.extend(batch_results)
    return results
```

## Key Changes Required

### In `video_analyzer/analyzer.py`:

1. Add async variant of `analyze_frame()`:
```python
async def analyze_frame_async(self, frame, previous_analyses=None):
    prompt = self.prompt_loader.get_by_index(0)
    if previous_analyses is not None:
        formatted = self._format_previous_analyses(previous_analyses)
    else:
        formatted = "No previous frames analyzed."

    prompt = prompt.replace("{PREVIOUS_FRAMES}", formatted)
    prompt = prompt.replace("{prompt}", self.user_prompt)

    # Use async client
    response = await self.llm_client.generate_async(
        prompt=prompt,
        image_path=frame.path,
        stream=False
    )
    return response['response']
```

2. Add `generate_async()` to LLM clients or wrap sync calls with `asyncio.to_thread()`:
```python
async def generate_async(self, prompt, image_path=None, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.generate(prompt, image_path, **kwargs)
    )
```

### In `video_analyzer/cli.py`:

Replace the sequential loop:
```python
# BEFORE
for frame in frames:
    analysis = analyzer.analyze_frame(frame, previous_analyses)
    frame_analyses.append(analysis)
    previous_analyses.append(analysis)
```

With async parallel processing:
```python
# AFTER
async def run_stage_2(analyzer, frames, config):
    mode = config.get("analysis_mode", "sliding-window")
    concurrency = config.get("concurrency", 4)
    window_size = config.get("window_size", 3)

    if mode == "parallel-independent":
        return await analyze_frames_parallel(analyzer, frames, concurrency)
    elif mode == "sliding-window":
        return await analyze_frames_sliding_window(analyzer, frames, window_size)
    # ... etc

frame_analyses = asyncio.run(run_stage_2(analyzer, frames, config))
```

## Performance Expectations

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| 30 frames, cloud API | 60s | 15s | 4x |
| 30 frames, local Ollama | 120s | 40s | 3x |
| 100 frames, cloud API | 200s | 25s | 8x |
| 100 frames, local GPU | 400s | 100s | 4x |

## Constraints & Warnings

- **Ollama concurrency**: Ollama queues parallel requests; high concurrency may not linearly speed up. Monitor `ollama ps`.
- **Rate limits**: Cloud APIs have rate limits. Add retry/backoff and respect `Retry-After` headers.
- **Context limits**: Sliding window prevents prompt length explosion but may lose distant context. Tune `window_size` per video type.
- **Memory**: Parallel analysis holds multiple frames/responses in memory. For very long videos, process in chunks.

## Verification

- [ ] Frame analysis time improves by at least 2x with `concurrency >= 2`
- [ ] Results quality is preserved (compare sequential vs parallel output)
- [ ] No rate limit errors with default concurrency
- [ ] Progress reporting still works (if implemented)
