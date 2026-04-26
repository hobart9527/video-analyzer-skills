# optimize-video-analysis

Analyze and optimize the video-analyzer codebase for performance, memory efficiency, and scalability.

## Description

Comprehensive optimization skill for the video-analyzer project (https://github.com/hobart9527/video-analyzer). Inspects the codebase, identifies bottlenecks, and applies targeted improvements across frame extraction, audio transcription, LLM inference, and pipeline orchestration.

## When to use

- When working on the video-analyzer project and need performance improvements
- Before processing long videos or batch video jobs
- When setting up the project on new hardware (GPU/CPU)
- When experiencing OOM errors or slow processing times

## Parameters

- target: Specific area to optimize ("all", "frame-extraction", "audio", "llm-client", "pipeline", "config")
- aggressive: Whether to apply aggressive optimizations (default: false)
- preserve-api: Whether to preserve existing public APIs (default: true)

## Instructions

1. Read and analyze the current codebase state
2. Identify the most impactful optimization based on the `target` parameter
3. Apply changes following the optimization patterns below
4. Verify the changes don't break existing functionality
5. Report improvements with before/after estimates

## Optimization Patterns

### 1. Frame Extraction (frame.py)

**Problem**: Full-frame memory caching, fixed threshold, redundant sampling

**Optimizations**:
- Replace `frame_candidates` list storing full `np.ndarray` with metadata-only storage (frame_number, score, timestamp)
- Implement single-pass top-K selection using `heapq.nlargest` instead of sort-then-truncate
- Add adaptive threshold based on frame brightness/contrast histogram
- Use `cv2.imencode` + memory buffer instead of `cv2.imwrite` for intermediate frames
- Add `max_memory_mb` parameter to limit RAM usage

**Key change locations**:
- `VideoProcessor.extract_keyframes()`: streaming extraction
- `VideoProcessor._calculate_frame_difference()`: multi-frame comparison

### 2. Audio Processing (audio_processor.py)

**Problem**: CPU-only, sync I/O, no batching, full audio loaded in memory

**Optimizations**:
- Auto-detect GPU and use `"float16"` or `"int8"` compute_type
- Replace `subprocess.run` with `asyncio.create_subprocess_exec`
- Add `torch.cuda.is_available()` / `torch.backends.mps.is_available()` detection
- Implement audio chunking for long videos (>30 min)
- Add confidence threshold filtering for low-quality segments
- Cache transcription results keyed by file hash

**Key change locations**:
- `AudioProcessor.__init__()`: device and compute_type selection
- `AudioProcessor.extract_audio()`: async subprocess
- `AudioProcessor.transcribe()`: chunked processing with progress callback

### 3. LLM Clients (clients/)

**Problem**: Sync requests, no connection reuse, string concatenation O(n^2), no timeout

**Optimizations**:
- Convert `OllamaClient` and `GenericOpenAIAPIClient` to async using `httpx`
- Add `requests.Session()` for sync path connection pooling
- Replace `accumulated_response += chunk` with `list` buffer + `"".join()`
- Add configurable timeouts: `timeout=(5, 60)` for connect/read
- Implement image encoding LRU cache keyed by `(path, mtime, size)`
- Add retry with exponential backoff in `OllamaClient`
- Implement batch frame analysis API when supported

**Key change locations**:
- `OllamaClient.generate()`: async + retry
- `GenericOpenAIAPIClient.generate()`: session reuse + SSE parsing fix
- `LLMClient.encode_image()`: LRU cache + resize

### 4. Video Analysis Pipeline (analyzer.py + cli.py)

**Problem**: Sequential frame analysis, linear context growth, no early stopping

**Optimizations**:
- Parallelize independent frame analysis using `asyncio.gather()` or `ThreadPoolExecutor`
- Implement sliding window for `previous_analyses` (keep last N frames or time-based sampling)
- Add similarity-based deduplication before sending to LLM
- Support incremental/streaming output for long videos
- Add progress callback hooks for UI integration
- Implement stage result caching (skip completed stages on resume)

**Key change locations**:
- `VideoAnalyzer.analyze_frame()`: sliding window context
- `cli.py main()`: parallel stage 1 (audio + frames concurrently)
- Pipeline: async frame analysis loop

### 5. Configuration (config.py)

**Problem**: pkg_resources deprecation, no caching, hardcoded mappings

**Optimizations**:
- Replace `pkg_resources` with `importlib.resources.files()`
- Add `@functools.lru_cache` to `load_config()`
- Extract `ARG_MAP` registry from `update_from_args()` hardcoded chain
- Use `pathlib.Path` consistently instead of string concatenation
- Add atomic file writes in `save_user_config()`
- Support environment variable overrides (`VIDEO_ANALYZER_*`)

**Key change locations**:
- `Config.__init__()`: importlib.resources
- `Config.load_config()`: caching
- `Config.update_from_args()`: registry pattern

### 6. Prompt Management (prompt.py)

**Problem**: No caching, repeated I/O, exception-as-control-flow

**Optimizations**:
- Add `@lru_cache` to `get_by_index()` and `get_by_name()`
- Pre-scan prompt directory at init to build `name -> path` index
- Use `importlib.resources` instead of `pkg_resources`
- Cache prompt templates after first load

**Key change locations**:
- `PromptLoader._find_prompt_file()`: path index
- `PromptLoader.get_by_index/name()`: LRU cache

## Verification Checklist

After applying optimizations:
- [ ] Frame extraction runs without OOM on 1-hour video
- [ ] Audio transcription detects and uses GPU if available
- [ ] LLM clients handle 429/503 with retry
- [ ] Frame analysis completes in <50% of original time (with parallelization)
- [ ] All existing CLI arguments still work
- [ ] Configuration saves/loads without errors
- [ ] Unit tests pass (if any exist)

## Example Usage

```
/optimize-video-analysis target=frame-extraction
/optimize-video-analysis target=all aggressive=true
/optimize-video-analysis target=llm-client preserve-api=false
```
