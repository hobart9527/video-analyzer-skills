# parallel-frame-analysis

## 概述

将 video-analyzer 中串行的逐帧分析循环替换为并行/批处理，显著缩短帧分析阶段耗时。支持独立并行、滑动窗口和 API 批量三种模式。

## 适用场景

- 关键帧数量较多（> 20 帧）
- 使用支持高并发的云端 LLM API
- 本地 GPU 有闲置算力可并行推理
- 批量视频处理任务

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| concurrency | int | 4 | 最大并行请求数（上限 16） |
| mode | string | "sliding-window" | 并行模式："parallel-independent" / "sliding-window" / "batch-api" |
| window_size | int | 3 | 滑动窗口中保留的上下文帧数 |
| provider | string | "auto" | LLM 提供商："ollama" / "openai" / "auto" |

## 核心指令

将 video-analyzer 中的串行帧分析循环 `for frame in frames: analyzer.analyze_frame(frame)` 替换为并行处理。

### 模式一：parallel-independent（独立并行）

适用于帧间无依赖关系的分析场景。使用 `asyncio.Semaphore` + `asyncio.gather()` 控制并发：

```python
async def analyze_frames_parallel(analyzer, frames, max_concurrency=4):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(frame):
        async with semaphore:
            return await analyzer.analyze_frame_async(frame)

    tasks = [analyze_one(f) for f in frames]
    return await asyncio.gather(*tasks)
```

### 模式二：sliding-window（滑动窗口）

适用于需要上下文依赖的分析。将 `previous_analyses` 限制为最近 N 帧，防止提示词长度爆炸：

```python
async def analyze_frames_sliding_window(analyzer, frames, window_size=3):
    results = []
    for i, frame in enumerate(frames):
        recent_context = results[-window_size:] if i > 0 else []
        result = await analyzer.analyze_frame_async(
            frame, previous_analyses=recent_context
        )
        results.append(result)
    return results
```

### 模式三：batch-api（API 批量）

适用于 OpenAI 兼容 API。将多帧打包为单个请求：

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

## 实现要点

1. 在 `video_analyzer/analyzer.py` 中添加异步变体 `analyze_frame_async()`：

```python
async def analyze_frame_async(self, frame, previous_analyses=None):
    prompt = self.prompt_loader.get_by_index(0)
    if previous_analyses is not None:
        formatted = self._format_previous_analyses(previous_analyses)
    else:
        formatted = "No previous frames analyzed."

    prompt = prompt.replace("{PREVIOUS_FRAMES}", formatted)
    prompt = prompt.replace("{prompt}", self.user_prompt)

    response = await self.llm_client.generate_async(
        prompt=prompt, image_path=frame.path, stream=False
    )
    return response['response']
```

2. 为 LLM 客户端添加 `generate_async()`，或使用 `asyncio.to_thread()` 包装同步调用。

3. 在 `video_analyzer/cli.py` 中替换串行循环为异步并行处理，使用 `asyncio.run()` 运行。

## 验证清单

- [ ] `concurrency >= 2` 时帧分析时间至少提升 2 倍
- [ ] 并行与串行输出质量一致（可对比验证）
- [ ] 默认并发下不触发 API 速率限制错误
- [ ] 进度报告正常工作（如已实现）

## 注意事项

- **Ollama 并发**：Ollama 会队列化并行请求，高并发不一定线性加速。建议通过 `ollama ps` 监控实际 GPU 利用率。
- **速率限制**：云端 API 有速率限制，需添加重试/退避逻辑并尊重 `Retry-After` 响应头。
- **上下文丢失**：滑动窗口可防止提示词长度爆炸，但可能丢失远距离上下文。建议根据视频类型调整 `window_size`。
- **内存**：并行分析会同时持有多个帧和响应结果，超长视频建议分块处理。

## 示例用法

```
/parallel-frame-analysis concurrency=4 mode=sliding-window
/parallel-frame-analysis concurrency=8 mode=parallel-independent provider=openai
/parallel-frame-analysis mode=batch-api window_size=5
```
