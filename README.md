# Video Analyzer 优化技能集

基于对 [video-analyzer](https://github.com/hobart9527/video-analyzer) 的深度代码分析，提取出 **8 个核心优化技能**，适配 **Claude Code**、**Codex** 和 **OpenClaw** 三种 AI 编程助手。

---

## 安装

### pip 安装（推荐）

```bash
pip install video-analyzer-skills
```

安装后可通过 `vas` 命令管理技能。

### 从源码安装

```bash
git clone https://github.com/hobart9527/video-analyzer-skills.git
cd video-analyzer-skills
pip install -e .
```

---

## CLI 用法

### 列出所有技能

```bash
vas list
```

### 查看单个技能

```bash
vas show gpu-auto-config
```

### 安装技能到目标平台

```bash
# 安装到 Claude Code（全局）
vas install --target claude-code

# 安装到 Codex（全局）
vas install --target codex

# 安装到 OpenClaw（项目级）
vas install --target openclaw

# 安装到所有平台
vas install --target all

# 项目级安装（安装到当前目录的 .claude/skills、.codex、.openclaw）
vas install --target claude-code --project

# 安装到指定目录
vas install --target claude-code --dest /path/to/video-analyzer/.claude/skills

# 只安装指定技能
vas install --target claude-code gpu-auto-config parallel-frame-analysis
```

### 检查环境

```bash
vas doctor
```

### 验证技能同步

```bash
vas render --check
```

---

## 原项目介绍

### 什么是 video-analyzer？

video-analyzer 是一款基于视觉大模型的视频分析工具，它结合 Llama 3.2 Vision 等视觉模型和 OpenAI Whisper 语音模型，通过提取视频关键帧并逐帧送入视觉模型分析，最终整合帧级分析结果与语音转录文本，生成对视频内容的自然语言描述。

### 核心功能

- **完全本地运行**：无需云服务或 API 密钥，可纯离线使用
- **云端加速**：支持任何 OpenAI 兼容 API（OpenRouter、OpenAI 等），兼顾速度与规模
- **智能关键帧提取**：基于 OpenCV 从视频中提取最具代表性的帧
- **高质量语音转录**：使用 Whisper 模型提取音频文字内容
- **逐帧视觉分析**：通过 Ollama 调用 Llama3.2 11B Vision 模型分析每一帧
- **自然语言视频描述**：输出连贯的视频内容描述文本
- **自动音频质量处理**：低质量音频自动降级处理
- **详细 JSON 输出**：包含元数据、转录、帧分析和最终描述
- **高度可配置**：支持命令行参数和配置文件双重配置

### 系统架构

video-analyzer 采用三阶段流水线架构：

**第一阶段：帧提取与音频处理**
- 使用 OpenCV 提取视频关键帧
- 使用 Whisper 进行语音转录
- 低置信度音频自动降级处理

**第二阶段：帧级分析**
- 逐帧送入视觉大模型分析
- 每帧分析包含前面帧的上下文信息
- 保持时序递进关系
- 使用 `frame_analysis.txt` 提示词模板

**第三阶段：视频重建**
- 按时间顺序整合所有帧分析结果
- 融合音频转录文本
- 使用首帧设定场景
- 生成完整的视频描述

![架构图](docs/design.png)

### 系统要求

- Python 3.11 或更高版本
- FFmpeg（音频处理必需）
- 本地运行 LLM 时（使用云端 API 则不需要）：
  - 至少 16GB 内存（推荐 32GB）
  - GPU 显存至少 12GB，或 Apple M 系列芯片配 32GB 统一内存

### 快速开始

```bash
# 本地运行（默认，使用 Ollama）
video-analyzer video.mp4

# 云端运行（OpenRouter）
video-analyzer video.mp4 \
    --client openai_api \
    --api-key your-key \
    --api-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.2-11b-vision-instruct:free
```

更多用法请参考原项目 [docs/USAGES.md](https://github.com/hobart9527/video-analyzer/blob/main/docs/USAGES.md)。

---

## 代码分析发现的主要瓶颈

通过对 video-analyzer 源码的静态分析，发现以下 7 个核心性能瓶颈：

| 模块 | 核心问题 | 影响 |
|------|----------|------|
| `frame.py` | 完整帧图像内存缓存、固定阈值 10.0、仅与前一帧比较 | 长视频 OOM、暗光/强光场景失效、渐变场景漏检 |
| `audio_processor.py` | 强制 float32/CPU、同步 I/O、无分块 | GPU 闲置、长视频内存高 |
| `clients/` | 同步 requests、无连接复用、O(n²) 字符串拼接 | 并发差、网络开销大 |
| `analyzer.py` | 完全串行帧分析、线性增长的上下文历史 | 处理时间长、可能超出上下文限制 |
| `cli.py` | 音视频串行执行、无进度回调 | 资源利用率低 |
| `config.py` | pkg_resources 已弃用、无缓存、硬编码映射 | 启动慢、维护难 |
| `prompt.py` | 无缓存、重复 I/O、异常作为控制流 | 重复文件读取 |

---

## 技能清单

### 1. optimize-video-analysis — 综合优化
**功能**：一站式分析并优化整个代码库  
**解决**：所有瓶颈的综合治理  
**适用**：首次优化、全面性能调优  
**调用**：`/optimize-video-analysis target=all`

### 2. parallel-frame-analysis — 并行帧分析
**功能**：将串行帧分析改为并行/批处理  
**解决**：帧分析是最大瓶颈（逐帧 LLM 调用）  
**适用**：关键帧数量多（>20）、使用云 API 时  
**预期提速**：2-8 倍（取决于并发度和 API 限制）  
**调用**：`/parallel-frame-analysis concurrency=4 mode=sliding-window`

### 3. adaptive-keyframe-extraction — 自适应关键帧提取
**功能**：智能阈值、内存流式提取、多帧差异检测  
**解决**：OOM、固定阈值失效、渐变场景漏检  
**适用**：长视频、内容变化大的视频  
**调用**：`/adaptive-keyframe-extraction strategy=adaptive-threshold`

### 4. async-llm-client — 异步 LLM 客户端
**功能**：将 requests 改为 httpx 异步客户端  
**解决**：阻塞 I/O、无连接池、无超时/重试  
**适用**：并行分析、UI 集成、高并发场景  
**调用**：`/async-llm-client client=both max_connections=10`

### 5. gpu-auto-config — GPU 自动配置
**功能**：自动检测 CUDA/MPS/ROCm 并优化设置  
**解决**：默认 CPU 运行、GPU 闲置  
**适用**：新环境部署、性能异常排查  
**调用**：`/gpu-auto-config verbose=true`

### 6. memory-streaming — 流式内存处理
**功能**：分块处理视频、流式输出、内存监控  
**解决**：长视频 OOM、内存不可控  
**适用**：>30 分钟视频、低内存机器  
**调用**：`/memory-streaming chunk_duration_sec=300 max_ram_mb=1024`

### 7. smart-prompt-cache — 智能缓存
**功能**：三级缓存（内存 LRU + SQLite 磁盘缓存 + 配置缓存）  
**解决**：重复 I/O、重复 LLM 推理  
**适用**：重复运行、提示词调优、相似视频  
**调用**：`/smart-prompt-cache max_cache_mb=1024`

### 8. batch-video-process — 批量视频处理
**功能**：多视频批量处理、模型共享、断点续传  
**解决**：逐视频重复加载模型、无批量工作流  
**适用**：视频目录处理、定时任务  
**调用**：`/batch-video-process input_pattern="videos/*.mp4" workers=2`

---

## 手动安装（不使用 CLI）

### Claude Code

将 `claude-code/*.md` 文件复制到 video-analyzer 项目的 `.claude/skills/` 目录：

```bash
mkdir -p /path/to/video-analyzer/.claude/skills/
cp claude-code/*.md /path/to/video-analyzer/.claude/skills/
```

或全局安装：

```bash
mkdir -p ~/.claude/skills/
cp claude-code/*.md ~/.claude/skills/
```

**调用方式**：在 Claude Code 中通过 `/skill-name` 调用，例如：

```
/optimize-video-analysis target=all
/parallel-frame-analysis concurrency=4 mode=sliding-window
/gpu-auto-config
```

### Codex (OpenAI)

```bash
mkdir -p .codex
cp codex/skills.json .codex/
```

### OpenClaw

```bash
mkdir -p .openclaw
cp openclaw/skills.md .openclaw/
```

---

## 优化效果预期

| 优化项 | 典型改进 |
|--------|----------|
| 并行帧分析 | 2-8 倍提速 |
| 自适应关键帧 | 减少 50% 冗余帧，暗光场景改善 |
| 异步客户端 | 并发能力提升，网络延迟隐藏 |
| GPU 自动配置 | Whisper 提速 5-10 倍（GPU vs CPU） |
| 流式处理 | 支持 2 小时+ 视频在 8GB 内存运行 |
| 智能缓存 | 重复运行提速 50%+ |
| 批量处理 | 多视频场景减少 30%+ 总时间 |

---

## 推荐实施顺序

1. **先用 `optimize-video-analysis` 做全面评估**
2. **优先实施 `gpu-auto-config` 和 `parallel-frame-analysis`**（投入产出比最高）
3. **针对长视频场景添加 `memory-streaming`**
4. **批处理场景添加 `batch-video-process`**

---

## 文件结构

```
video-analyzer-skills/
├── README.md                           # 项目说明
├── pyproject.toml                      # Python 包配置
├── video_analyzer_skills/              # CLI 工具包
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                          # CLI 入口
│   ├── parser.py                       # 技能解析器
│   └── installer.py                    # 安装器
├── claude-code/                        # Claude Code 技能（8 个 .md）
│   ├── optimize-video-analysis.md
│   ├── parallel-frame-analysis.md
│   ├── adaptive-keyframe-extraction.md
│   ├── async-llm-client.md
│   ├── gpu-auto-config.md
│   ├── memory-streaming.md
│   ├── smart-prompt-cache.md
│   └── batch-video-process.md
├── codex/
│   └── skills.json                     # Codex 格式技能定义
├── openclaw/
│   └── skills.md                       # OpenClaw 格式技能定义
└── tests/                              # 测试
```

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
ruff check video_analyzer_skills/
```

---

## 许可证

Apache License 2.0
