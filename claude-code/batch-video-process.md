# batch-video-process

Process multiple videos efficiently with batching, shared model loading, and progress tracking.

## Description

Extends video-analyzer to handle multiple videos in a single invocation, sharing loaded models across videos and providing aggregated progress reporting. Eliminates redundant model loading and enables efficient batch workflows.

## When to use

- When processing a directory of videos
- When running video analysis as a scheduled job
- When comparing analysis results across multiple videos
- When the same models/configs apply to all videos

## Parameters

- input_pattern: Glob pattern for input videos (e.g., `"videos/*.mp4"`)
- output_dir: Base directory for all outputs
- workers: Number of parallel videos (default: 1, increase with GPU)
- skip_existing: Skip videos that already have output (default: true)
- aggregate: Generate summary across all videos (default: false)

## Implementation

### 1. Batch Orchestrator

```python
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict

@dataclass
class BatchResult:
    video_path: str
    status: str  # "success" | "error" | "skipped"
    output_path: Optional[str] = None
    duration_sec: Optional[float] = None
    num_keyframes: Optional[int] = None
    error_message: Optional[str] = None

class BatchVideoProcessor:
    def __init__(self, config: dict, max_workers: int = 1):
        self.config = config
        self.max_workers = max_workers
        self.results: List[BatchResult] = []

        # Shared model instances (loaded once)
        self._audio_processor = None
        self._llm_client = None
        self._analyzer = None

    def _init_shared_models(self):
        """Initialize models once for reuse across videos."""
        from video_analyzer.audio_processor import AudioProcessor
        from video_analyzer.config import get_client

        client_config = get_client(self.config)
        self._audio_processor = AudioProcessor(
            model_size=self.config.get("audio_model", "base"),
            device=self.config.get("device", "cpu"),
            compute_type=self.config.get("compute_type", "float32")
        )

        # Initialize LLM client based on config
        client_type = self.config.get("clients", {}).get("default", "ollama")
        if client_type == "ollama":
            from video_analyzer.clients.ollama import OllamaClient
            self._llm_client = OllamaClient(
                host=client_config.get("host", "http://localhost:11434"),
                model=client_config.get("model", "llama3.2-vision")
            )
        else:
            from video_analyzer.clients.generic_openai_api import GenericOpenAIAPIClient
            self._llm_client = GenericOpenAIAPIClient(**client_config)

    async def process_batch(self, video_paths: List[str], output_base: str,
                           skip_existing: bool = True,
                           progress_callback = None) -> List[BatchResult]:
        """Process multiple videos with shared models."""
        self._init_shared_models()
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_one(video_path: str) -> BatchResult:
            async with semaphore:
                return await self._process_single(
                    video_path, output_base, skip_existing, progress_callback
                )

        tasks = [process_one(vp) for vp in video_paths]
        self.results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed = []
        for path, result in zip(video_paths, self.results):
            if isinstance(result, Exception):
                processed.append(BatchResult(
                    video_path=path,
                    status="error",
                    error_message=str(result)
                ))
            else:
                processed.append(result)

        self.results = processed
        return processed

    async def _process_single(self, video_path: str, output_base: Path,
                              skip_existing: bool,
                              progress_callback) -> BatchResult:
        video_name = Path(video_path).stem
        output_dir = output_base / video_name
        result_file = output_dir / "analysis.json"

        if skip_existing and result_file.exists():
            return BatchResult(
                video_path=video_path,
                status="skipped",
                output_path=str(result_file)
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use existing pipeline but with shared models
            from video_analyzer.frame import VideoProcessor
            from video_analyzer.analyzer import VideoAnalyzer
            from video_analyzer.prompt import PromptLoader

            # Stage 1: Extract frames and audio (concurrently)
            frame_extractor = VideoProcessor(video_path, str(output_dir / "frames"))
            frames = frame_extractor.extract_keyframes(
                target_frames=self.config.get("max_frames", 10)
            )

            audio_result = self._audio_processor.transcribe(video_path)

            # Stage 2: Analyze frames
            prompt_loader = PromptLoader(
                self.config.get("prompt_dir"),
                self.config.get("prompts", [])
            )
            analyzer = VideoAnalyzer(
                llm_client=self._llm_client,
                model=self.config.get("model"),
                prompt_loader=prompt_loader,
                user_prompt=self.config.get("user_prompt", "")
            )

            previous_analyses = []
            frame_analyses = []
            for frame in frames:
                analysis = analyzer.analyze_frame(frame, previous_analyses)
                frame_analyses.append(analysis)
                previous_analyses.append(analysis)

            # Stage 3: Reconstruct
            final_description = analyzer.reconstruct_video(
                frame_analyses=frame_analyses,
                audio_transcript=audio_result,
                first_frame=frames[0] if frames else None
            )

            # Save results
            result = {
                "video": video_path,
                "description": final_description,
                "frames": [
                    {"number": f.frame_number, "timestamp": f.timestamp,
                     "analysis": a}
                    for f, a in zip(frames, frame_analyses)
                ],
                "audio": {
                    "text": audio_result.text if hasattr(audio_result, 'text') else str(audio_result),
                    "language": getattr(audio_result, 'language', 'unknown')
                }
            }

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            if progress_callback:
                progress_callback(video_path, "complete")

            return BatchResult(
                video_path=video_path,
                status="success",
                output_path=str(result_file),
                num_keyframes=len(frames)
            )

        except Exception as e:
            if progress_callback:
                progress_callback(video_path, f"error: {e}")
            return BatchResult(
                video_path=video_path,
                status="error",
                error_message=str(e)
            )

    def generate_report(self, output_path: str):
        """Generate aggregate report of batch processing."""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "success")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        errors = total - success - skipped

        report = {
            "summary": {
                "total_videos": total,
                "successful": success,
                "skipped": skipped,
                "errors": errors,
            },
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
```

### 2. CLI Extension

```python
def add_batch_arguments(parser):
    """Add batch processing arguments to CLI."""
    batch_group = parser.add_argument_group("Batch Processing")
    batch_group.add_argument("--batch", action="store_true",
                             help="Process multiple videos")
    batch_group.add_argument("--input-pattern", type=str,
                             help="Glob pattern for input videos")
    batch_group.add_argument("--batch-workers", type=int, default=1,
                             help="Number of parallel videos")
    batch_group.add_argument("--skip-existing", action="store_true", default=True,
                             help="Skip videos with existing output")
    batch_group.add_argument("--aggregate", action="store_true",
                             help="Generate aggregate summary report")

async def run_batch(args, config):
    """Execute batch processing."""
    import glob

    video_paths = glob.glob(args.input_pattern)
    if not video_paths:
        print(f"No videos found matching: {args.input_pattern}")
        return

    print(f"Found {len(video_paths)} videos to process")

    processor = BatchVideoProcessor(config, max_workers=args.batch_workers)

    def progress(path, status):
        print(f"  [{status}] {Path(path).name}")

    results = await processor.process_batch(
        video_paths,
        args.output_dir,
        skip_existing=args.skip_existing,
        progress_callback=progress
    )

    # Print summary
    success = sum(1 for r in results if r.status == "success")
    print(f"\nComplete: {success}/{len(results)} videos processed successfully")

    if args.aggregate:
        report_path = Path(args.output_dir) / "batch_report.json"
        processor.generate_report(str(report_path))
        print(f"Aggregate report saved to: {report_path}")
```

### 3. Resume Capability

```python
class ResumableBatch:
    """Batch processing that can resume after interruption."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.completed = set()
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.completed = set(data.get("completed", []))

    def _save_state(self):
        self.state_file.write_text(json.dumps({
            "completed": list(self.completed)
        }))

    def is_completed(self, video_path: str) -> bool:
        return video_path in self.completed

    def mark_completed(self, video_path: str):
        self.completed.add(video_path)
        self._save_state()

    def get_pending(self, all_paths: List[str]) -> List[str]:
        return [p for p in all_paths if not self.is_completed(p)]
```

## Verification

- [ ] Multiple videos process in a single invocation
- [ ] Models are loaded only once (check process memory)
- [ ] Existing outputs are skipped when `skip_existing=true`
- [ ] Aggregate report contains all video results
- [ ] Interrupted batch can resume from state file
- [ ] Error in one video doesn't stop entire batch
