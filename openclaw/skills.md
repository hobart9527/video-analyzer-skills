# Video Analyzer Optimization Skills

Collection of skills for optimizing the video-analyzer project across frame extraction, audio processing, LLM inference, and pipeline orchestration.

---

## optimize-video-analysis

**Trigger**: When working on video-analyzer and need performance improvements

**Prompt**: Analyze the video-analyzer codebase and apply comprehensive optimizations. Focus areas: 1) Frame extraction - replace full-frame memory cache with metadata-only storage, add adaptive thresholds, use heapq for single-pass selection. 2) Audio processing - auto-detect GPU for Whisper, use float16/int8, async subprocess, chunked processing for long videos. 3) LLM clients - convert to async httpx, add connection pooling, replace O(n^2) string concat with list+join, add retry with backoff. 4) Pipeline - parallelize frame analysis with sliding window context, run audio+frames concurrently in Stage 1. 5) Config - replace pkg_resources with importlib.resources, add LRU cache, atomic writes. 6) Prompts - pre-build path index, cache loaded templates. Preserve existing CLI APIs.

**Parameters**:
- target: "all" | "frame-extraction" | "audio" | "llm-client" | "pipeline" | "config"
- aggressive: bool (default false)

---

## parallel-frame-analysis

**Trigger**: When frame analysis is bottleneck and video has many keyframes

**Prompt**: Convert sequential frame analysis to parallel processing. Implement three modes: 1) parallel-independent - asyncio.gather() with semaphore for independent frames. 2) sliding-window - limit previous_analyses to last N frames (default 3) to prevent prompt explosion while maintaining context. 3) batch-api - group frames into API batch requests for OpenAI-compatible endpoints. Add async analyze_frame_async() to VideoAnalyzer. Update cli.py to use asyncio.run(). Add concurrency config. Respect Ollama queue behavior and cloud API rate limits.

**Parameters**:
- concurrency: int (default 4, max 16)
- mode: "parallel-independent" | "sliding-window" | "batch-api"
- window_size: int (default 3)

---

## adaptive-keyframe-extraction

**Trigger**: When keyframes are poor quality or processing causes OOM

**Prompt**: Replace fixed-threshold keyframe extraction with adaptive, memory-efficient implementation. Changes: 1) StreamingKeyframeExtractor stores only (frame_number, score, timestamp) instead of full np.ndarray. 2) Adaptive threshold using median + sensitivity * std of recent scores. 3) Histogram-based difference option for lighting invariance. 4) Multi-frame difference for gradual transition detection. 5) Temporal filtering with min_scene_duration_sec. Use heapq for single-pass top-K. Add strategy config: adaptive-threshold, histogram, multi-frame.

**Parameters**:
- strategy: "adaptive-threshold" | "histogram" | "multi-frame"
- max_memory_mb: float (default 512)
- min_scene_duration_sec: float (default 1.0)
- sensitivity: float (default 0.5)

---

## async-llm-client

**Trigger**: When implementing parallel frame analysis or integrating with async frameworks

**Prompt**: Convert sync requests-based LLM clients to async httpx. Create AsyncLLMClient base with AsyncOllamaClient and AsyncOpenAIClient. Features: connection pooling via httpx.AsyncClient, async image encoding with LRU cache, proper SSE parsing (strip 'data: ' prefix, handle [DONE]), list buffer + ''.join() for streaming, configurable timeouts. Add sync wrappers for backward compatibility. Implement retry with exponential backoff for 429/502/503/504 respecting Retry-After header. Update analyzer.py and cli.py for async pipeline.

**Parameters**:
- client: "ollama" | "openai" | "both"
- max_connections: int (default 10)
- timeout: float (default 60)

---

## gpu-auto-config

**Trigger**: When setting up video-analyzer on new hardware or performance is unexpectedly slow

**Prompt**: Auto-detect hardware and optimize settings. Detection priority: CUDA (torch.cuda) -> MPS (Apple Silicon) -> ROCm -> CPU. For CUDA, check compute capability for float16 vs int8. For CPU, check AVX2 support. Update AudioProcessor to use detected device and compute_type. Add get_optimal_workers() based on hardware. Detect Ollama GPU via 'ollama ps'. Print hardware summary on startup. Add apply_hardware_defaults() to Config. Support force_device override.

**Parameters**:
- force_device: "cuda" | "mps" | "cpu" (optional override)
- whisper_compute_type: "float16" | "int8" | "float32" (optional override)
- verbose: bool (default true)

---

## memory-streaming

**Trigger**: When processing long videos or encountering OOM errors

**Prompt**: Implement chunked, streaming processing for memory efficiency. Components: 1) ChunkedVideoProcessor splits video into chunk_duration_sec segments, processes independently, yields metadata. 2) Streaming keyframe extraction per-chunk with distributed target_frames. 3) Streaming audio transcription using soundfile to read chunks, transcribe independently, adjust timestamps. 4) Streaming analysis results flushed to JSONL every N frames. 5) MemoryMonitor checks RSS periodically, triggers gc.collect(), raises MemoryError with helpful message if limit exceeded. Add --chunk-duration, --max-ram-mb, --stream-output CLI options.

**Parameters**:
- chunk_duration_sec: float (default 300)
- max_ram_mb: float (default 1024)
- cleanup_interval: int (default 100)

---

## smart-prompt-cache

**Trigger**: When re-running analysis, tuning prompts, or processing similar videos

**Prompt**: Add multi-level caching. 1) CachedPromptLoader with @lru_cache on get_by_index/name, pre-built path index at init, importlib.resources instead of pkg_resources. 2) Disk cache via SQLite for analysis results: key=SHA256(prompt+image_path+mtime+model+temperature), value=pickled result, TTL expiration, LRU size cleanup. Thread-safe for parallel access. 3) CachedLLMClient wrapper that skips cache for streaming. 4) File-change-aware config cache checking mtime. Add cache stats and clear command.

**Parameters**:
- cache_dir: str (default ~/.cache/video-analyzer)
- max_cache_mb: float (default 1024)
- cache_ttl_hours: float (default 168)

---

## batch-video-process

**Trigger**: When processing directories of videos or running scheduled analysis jobs

**Prompt**: Add batch processing with shared models. BatchVideoProcessor loads AudioProcessor and LLM client ONCE, processes videos with semaphore-controlled parallelism. Features: skip_existing checks, progress callbacks, per-video BatchResult tracking, aggregate JSON report, resumable state file for interruptions. Error in one video doesn't stop batch. Add --batch, --input-pattern, --batch-workers, --skip-existing, --aggregate CLI options.

**Parameters**:
- input_pattern: str (glob pattern)
- output_dir: str
- workers: int (default 1)
- skip_existing: bool (default true)
- aggregate: bool (default false)
