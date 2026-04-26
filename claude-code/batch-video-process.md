# batch-video-process

## 概述

为 video-analyzer 添加批量视频处理能力，支持多视频单次调用、模型共享加载和进度追踪。消除逐视频重复加载模型开销，实现高效的批量工作流。

## 适用场景

- 处理目录中的多个视频文件
- 将视频分析作为定时任务运行
- 跨多视频对比分析结果
- 所有视频使用相同模型和配置

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| input_pattern | string | 必填 | 输入视频 glob 模式（如 `"videos/*.mp4"`） |
| output_dir | string | "./output" | 输出目录基准路径 |
| workers | int | 1 | 并行处理视频数（GPU 环境可适当增加） |
| skip_existing | bool | true | 跳过已有输出结果的视频 |
| aggregate | bool | false | 是否生成批量汇总报告 |

## 核心指令

实现批量视频处理流水线，共享模型实例并支持断点续传。

### 1. 批量编排器

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
        self._audio_processor = None
        self._llm_client = None
        self._analyzer = None

    def _init_shared_models(self):
        """初始化模型（仅一次，跨视频复用）。"""
        from video_analyzer.audio_processor import AudioProcessor
        from video_analyzer.config import get_client

        client_config = get_client(self.config)
        self._audio_processor = AudioProcessor(
            model_size=self.config.get("audio_model", "base"),
            device=self.config.get("device", "cpu"),
            compute_type=self.config.get("compute_type", "float32")
        )
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
                           progress_callback=None) -> List[BatchResult]:
        """使用共享模型处理多个视频。"""
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
        processed = []
        for path, result in zip(video_paths, self.results):
            if isinstance(result, Exception):
                processed.append(BatchResult(
                    video_path=path, status="error", error_message=str(result)
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
                video_path=video_path, status="skipped",
                output_path=str(result_file)
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from video_analyzer.frame import VideoProcessor
            from video_analyzer.analyzer import VideoAnalyzer
            from video_analyzer.prompt import PromptLoader

            frame_extractor = VideoProcessor(video_path, str(output_dir / "frames"))
            frames = frame_extractor.extract_keyframes(
                target_frames=self.config.get("max_frames", 10)
            )
            audio_result = self._audio_processor.transcribe(video_path)
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

            final_description = analyzer.reconstruct_video(
                frame_analyses=frame_analyses,
                audio_transcript=audio_result,
                first_frame=frames[0] if frames else None
            )
            result = {
                "video": video_path,
                "description": final_description,
                "frames": [
                    {"number": f.frame_number, "timestamp": f.timestamp, "analysis": a}
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
                video_path=video_path, status="success",
                output_path=str(result_file), num_keyframes=len(frames)
            )
        except Exception as e:
            if progress_callback:
                progress_callback(video_path, f"error: {e}")
            return BatchResult(
                video_path=video_path, status="error", error_message=str(e)
            )

    def generate_report(self, output_path: str):
        """生成批量处理汇总报告。"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "success")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        errors = total - success - skipped
        report = {
            "summary": {
                "total_videos": total, "successful": success,
                "skipped": skipped, "errors": errors,
            },
            "results": [asdict(r) for r in self.results],
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
```

### 2. CLI 扩展

```python
def add_batch_arguments(parser):
    """为 CLI 添加批量处理参数。"""
    batch_group = parser.add_argument_group("批量处理")
    batch_group.add_argument("--batch", action="store_true",
                             help="启用批量视频处理模式")
    batch_group.add_argument("--input-pattern", type=str,
                             help="输入视频的 glob 匹配模式")
    batch_group.add_argument("--batch-workers", type=int, default=1,
                             help="并行处理视频数")
    batch_group.add_argument("--skip-existing", action="store_true", default=True,
                             help="跳过已有输出结果的视频")
    batch_group.add_argument("--aggregate", action="store_true",
                             help="生成批量汇总报告")

async def run_batch(args, config):
    """执行批量处理。"""
    import glob
    video_paths = glob.glob(args.input_pattern)
    if not video_paths:
        print(f"未找到匹配的视频: {args.input_pattern}")
        return
    print(f"找到 {len(video_paths)} 个待处理视频")
    processor = BatchVideoProcessor(config, max_workers=args.batch_workers)
    def progress(path, status):
        print(f"  [{status}] {Path(path).name}")
    results = await processor.process_batch(
        video_paths, args.output_dir,
        skip_existing=args.skip_existing, progress_callback=progress
    )
    success = sum(1 for r in results if r.status == "success")
    print(f"\n完成: {success}/{len(results)} 个视频处理成功")
    if args.aggregate:
        report_path = Path(args.output_dir) / "batch_report.json"
        processor.generate_report(str(report_path))
        print(f"汇总报告已保存至: {report_path}")
```

### 3. 断点续传

```python
class ResumableBatch:
    """支持中断后恢复处理的批量任务。"""
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.completed = set()
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.completed = set(data.get("completed", []))

    def _save_state(self):
        self.state_file.write_text(json.dumps({"completed": list(self.completed)}))

    def is_completed(self, video_path: str) -> bool:
        return video_path in self.completed

    def mark_completed(self, video_path: str):
        self.completed.add(video_path)
        self._save_state()

    def get_pending(self, all_paths: List[str]) -> List[str]:
        return [p for p in all_paths if not self.is_completed(p)]
```

## 实现要点

- `BatchVideoProcessor` 在初始化时仅加载一次 `AudioProcessor` 和 LLM 客户端
- 使用 `asyncio.Semaphore` 控制并发度，避免资源耗尽
- 单个视频出错不会中断整个批量任务
- `--skip-existing` 根据输出文件存在性判断，支持幂等重跑
- 断点续传状态保存在 `batch_state.json` 中

## 验证清单

- [ ] 单次调用处理多个视频
- [ ] 模型仅加载一次（通过内存占用验证）
- [ ] `skip_existing=true` 时正确跳过已有输出
- [ ] 汇总报告包含所有视频结果
- [ ] 中断后可从状态文件恢复处理
- [ ] 单个视频错误不终止整个批量任务

## 示例用法

```
/batch-video-process input_pattern="videos/*.mp4" workers=2
/batch-video-process input_pattern="~/Downloads/*.mp4" skip_existing=true aggregate=true
/batch-video-process input_pattern="dataset/**/*.mov" workers=4
```
