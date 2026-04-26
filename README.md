# Video Analyzer Optimization Skills

基于对 [hobart9527/video-analyzer](https://github.com/hobart9527/video-analyzer) 的深度代码分析，提取出的 **8个可优化功能点**，转化为适用于 **Claude Code**、**Codex** 和 **OpenClaw** 的 Skill 定义。

---

## 项目分析摘要

### 核心架构
video-analyzer 是一个三阶段视频分析流水线：
1. **音视频提取** — OpenCV提取关键帧 + Whisper转录音频
2. **帧级分析** — 视觉LLM逐帧分析（带历史上下文）
3. **视频重建** — 时序整合生成完整描述

### 发现的主要瓶颈

| 模块 | 核心问题 | 影响 |
|------|----------|------|
| `frame.py` | 完整帧图像内存缓存、固定阈值、仅与前一帧比较 | 长视频OOM、暗光/亮光场景失效、渐变场景漏检 |
| `audio_processor.py` | 强制float32/CPU、同步I/O、无分块 | GPU闲置、长视频内存高 |
| `clients/` | 同步requests、无连接复用、O(n²)字符串拼接 | 并发差、网络开销大 |
| `analyzer.py` | 完全串行帧分析、线性增长的上下文历史 | 处理时间长、可能超出上下文限制 |
| `cli.py` | 音视频串行执行、无进度回调 | 资源利用率低 |
| `config.py` | pkg_resources已弃用、无缓存、硬编码映射 | 启动慢、维护难 |
| `prompt.py` | 无缓存、重复I/O、异常作为控制流 | 重复文件读取 |

---

## Skill 清单

### 1. optimize-video-analysis（综合优化）
**功能**: 一站式分析并优化整个代码库  
**解决**: 所有上述瓶颈的综合治理  
**适用**: 首次优化、全面性能调优

### 2. parallel-frame-analysis（并行帧分析）
**功能**: 将串行帧分析改为并行/批处理  
**解决**: 帧分析是最大瓶颈（逐帧LLM调用）  
**适用**: 关键帧数量多（>20）、使用云API时  
**预期提速**: 2-8倍（取决于并发度和API限制）

### 3. adaptive-keyframe-extraction（自适应关键帧提取）
**功能**: 智能阈值、内存流式提取、多帧差异检测  
**解决**: OOM、固定阈值失效、渐变场景漏检  
**适用**: 长视频、内容变化大的视频

### 4. async-llm-client（异步LLM客户端）
**功能**: 将requests改为httpx异步客户端  
**解决**: 阻塞I/O、无连接池、无超时/重试  
**适用**: 并行分析、UI集成、高并发场景

### 5. gpu-auto-config（GPU自动配置）
**功能**: 自动检测CUDA/MPS/ROCm并优化设置  
**解决**: 默认CPU运行、GPU闲置  
**适用**: 新环境部署、性能异常排查

### 6. memory-streaming（流式内存处理）
**功能**: 分块处理视频、流式输出、内存监控  
**解决**: 长视频OOM、内存不可控  
**适用**: >30分钟视频、低内存机器

### 7. smart-prompt-cache（智能缓存）
**功能**: 三级缓存（内存LRU + SQLite磁盘缓存 + 配置缓存）  
**解决**: 重复I/O、重复LLM推理  
**适用**: 重复运行、提示词调优、相似视频

### 8. batch-video-process（批量视频处理）
**功能**: 多视频批量处理、模型共享、断点续传  
**解决**: 逐视频重复加载模型、无批量工作流  
**适用**: 视频目录处理、定时任务

---

## 安装方式

### Claude Code

将 `claude-code/*.md` 文件复制到项目的 `.claude/skills/` 目录：

```bash
mkdir -p /path/to/video-analyzer/.claude/skills/
cp claude-code/*.md /path/to/video-analyzer/.claude/skills/
```

或在用户全局目录：
```bash
mkdir -p ~/.claude/skills/
cp claude-code/*.md ~/.claude/skills/
```

然后在 Claude Code 中通过 `/skill-name` 调用，例如：
```
/optimize-video-analysis target=all
/parallel-frame-analysis concurrency=4 mode=sliding-window
/gpu-auto-config
```

### Codex (OpenAI)

将 `codex/skills.json` 的内容合并到你的 Codex 技能配置中：

```bash
# 如果有现有 skills.json，合并数组
jq -s '.[0].skills + .[1].skills | {skills: .}' \
  existing_skills.json codex/skills.json > merged_skills.json
```

或在 Codex 项目目录创建：
```bash
mkdir -p .codex
cp codex/skills.json .codex/
```

### OpenClaw

将 `openclaw/skills.md` 作为技能定义文档使用。根据 OpenClaw 的具体格式要求，可将每个技能段落转换为对应的技能配置文件。

---

## 优化效果预期

| 优化项 | 典型改进 |
|--------|----------|
| 并行帧分析 | 2-8倍提速 |
| 自适应关键帧 | 减少50%冗余帧，暗光场景改善 |
| 异步客户端 | 并发能力提升，网络延迟隐藏 |
| GPU自动配置 | Whisper提速5-10倍（GPU vs CPU） |
| 流式处理 | 支持2小时+视频在8GB内存运行 |
| 智能缓存 | 重复运行提速50%+ |
| 批量处理 | 多视频场景减少30%+总时间 |

---

## 文件结构

```
video-analyzer-skills/
├── README.md                          # 本文件
├── claude-code/
│   ├── optimize-video-analysis.md     # 综合优化
│   ├── parallel-frame-analysis.md     # 并行帧分析
│   ├── adaptive-keyframe-extraction.md # 自适应关键帧
│   ├── async-llm-client.md            # 异步LLM客户端
│   ├── gpu-auto-config.md             # GPU自动配置
│   ├── memory-streaming.md            # 流式内存处理
│   ├── smart-prompt-cache.md          # 智能缓存
│   └── batch-video-process.md         # 批量视频处理
├── codex/
│   └── skills.json                    # Codex 格式技能定义
└── openclaw/
    └── skills.md                      # OpenClaw 格式技能定义
```

---

## 贡献与扩展

这些技能基于对原项目代码的静态分析生成。实际应用时建议：
1. 先用 `optimize-video-analysis` 做全面评估
2. 优先实施 `gpu-auto-config` 和 `parallel-frame-analysis`（投入产出比最高）
3. 针对长视频场景添加 `memory-streaming`
4. 批处理场景添加 `batch-video-process`
