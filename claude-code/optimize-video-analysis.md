# optimize-video-analysis

## 概述

对 video-analyzer 代码库进行全面性能分析和优化，覆盖帧提取、音频转录、LLM 推理和流水线编排四大模块。通过一次调用即可评估并实施所有关键优化点。

## 适用场景

- 首次对 video-analyzer 进行性能优化
- 处理长视频或批量视频任务前
- 在新硬件（GPU/CPU）上部署项目
- 遇到 OOM 错误或处理速度过慢

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| target | string | "all" | 优化目标："all" / "frame-extraction" / "audio" / "llm-client" / "pipeline" / "config" |
| aggressive | bool | false | 是否启用激进优化（可能破坏 API 兼容性） |
| preserve-api | bool | true | 是否保留现有公共 API |

## 核心指令

分析 video-analyzer 代码库并应用针对性优化。按以下优先级执行：

1. **帧提取（frame.py）**：将 `frame_candidates` 中存储的完整 `np.ndarray` 替换为元数据存储（frame_number, score, timestamp），使用 `heapq.nlargest` 实现单遍 top-K 选择，添加自适应阈值（基于亮度/对比度直方图），引入 `max_memory_mb` 参数限制内存。

2. **音频处理（audio_processor.py）**：自动检测 GPU 并使用 `"float16"` 或 `"int8"` 计算类型，将 `subprocess.run` 替换为 `asyncio.create_subprocess_exec`，为长视频（>30 分钟）实现音频分块处理，按文件哈希缓存转录结果。

3. **LLM 客户端（clients/）**：将 `OllamaClient` 和 `GenericOpenAIAPIClient` 转换为基于 `httpx` 的异步实现，同步路径使用 `requests.Session()` 复用连接，将 `accumulated_response += chunk` 替换为 `list` 缓冲区 + `"".join()`，添加可配置超时 `timeout=(5, 60)`，实现指数退避重试。

4. **流水线（analyzer.py + cli.py）**：使用 `asyncio.gather()` 或 `ThreadPoolExecutor` 并行化独立帧分析，为 `previous_analyses` 实现滑动窗口（保留最近 N 帧），在第一阶段并发执行音频和帧提取，添加进度回调钩子。

5. **配置（config.py）**：将 `pkg_resources` 替换为 `importlib.resources.files()`，为 `load_config()` 添加 `@functools.lru_cache`，从 `update_from_args()` 中提取硬编码的 `ARG_MAP` 为注册表模式，使用 `pathlib.Path` 替代字符串拼接，添加原子文件写入。

6. **提示词管理（prompt.py）**：为 `get_by_index()` 和 `get_by_name()` 添加 `@lru_cache`，在初始化时预扫描提示词目录构建 `name -> path` 索引，使用 `importlib.resources` 替代 `pkg_resources`。

## 实现要点

- **关键修改位置**：
  - `VideoProcessor.extract_keyframes()` → 流式提取
  - `AudioProcessor.__init__()` → 设备和计算类型选择
  - `OllamaClient.generate()` → 异步 + 重试
  - `VideoAnalyzer.analyze_frame()` → 滑动窗口上下文
  - `Config.__init__()` → importlib.resources
  - `PromptLoader._find_prompt_file()` → 路径索引

- **依赖变更**：
  - 新增：`httpx`
  - 移除：`pkg_resources` 相关依赖

## 验证清单

- [ ] 1 小时视频的帧提取不触发 OOM
- [ ] 音频转录自动检测并使用可用 GPU
- [ ] LLM 客户端对 429/503 错误触发重试
- [ ] 并行化后帧分析时间缩短至原时长的 50% 以下
- [ ] 所有现有 CLI 参数仍然有效
- [ ] 配置保存/加载无错误
- [ ] 如有单元测试则全部通过

## 示例用法

```
/optimize-video-analysis target=frame-extraction
/optimize-video-analysis target=all aggressive=true
/optimize-video-analysis target=llm-client preserve-api=false
```
