# async-llm-client

## 概述

将 video-analyzer 中基于 `requests` 的同步 LLM 客户端替换为基于 `httpx` 的异步实现，支持并发 API 调用、连接池和响应式 I/O，为并行帧分析提供基础。

## 适用场景

- 已实现或计划实现并行帧分析
- 使用有速率限制的云端 API（异步 + 信号量 = 可控并发）
- 集成异步 Web 框架（FastAPI 等）
- UI 需要在分析期间保持响应

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| client | string | "both" | 转换目标："ollama" / "openai" / "both" |
| max_connections | int | 10 | 连接池大小 |
| timeout | float | 60 | 请求超时（秒） |
| preserve_sync | bool | true | 保留同步 API 作为包装器 |

## 核心指令

将同步 `requests` 客户端替换为异步 `httpx` 客户端，按以下步骤执行：

### 1. 创建异步基类

创建 `video_analyzer/clients/async_llm_client.py`：

```python
import base64
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import httpx

class AsyncLLMClient(ABC):
    def __init__(self, timeout: float = 60.0, max_connections: int = 10):
        self.timeout = timeout
        limits = httpx.Limits(max_connections=max_connections)
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)

    async def encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        async with asyncio.Lock():
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    @abstractmethod
    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        pass

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
```

### 2. 异步 Ollama 客户端

```python
import json
import httpx

class AsyncOllamaClient(AsyncLLMClient):
    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "llama3.2-vision", timeout: float = 120.0,
                 max_connections: int = 10):
        super().__init__(timeout=timeout, max_connections=max_connections)
        self.host = host.rstrip("/")
        self.model = model
        self.generate_url = f"{self.host}/api/generate"

    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": kwargs.get("num_predict", 300),
                "temperature": kwargs.get("temperature", 0.7),
            }
        }
        if image_path:
            data["images"] = [await self.encode_image(image_path)]

        response = await self.client.post(self.generate_url, json=data)
        response.raise_for_status()

        if stream:
            text_parts = []
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        text_parts.append(chunk.get("response", ""))
                    except json.JSONDecodeError:
                        continue
            return {"response": "".join(text_parts)}
        return response.json()
```

### 3. 异步 OpenAI 兼容客户端

```python
class AsyncOpenAIClient(AsyncLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o",
                 timeout: float = 60.0, max_connections: int = 10):
        super().__init__(timeout=timeout, max_connections=max_connections)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_url = f"{self.base_url}/chat/completions"

    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if image_path:
            encoded = await self.encode_image(image_path)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]

        data = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": kwargs.get("max_tokens", 300),
            "temperature": kwargs.get("temperature", 0.7),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post(self.chat_url, json=data, headers=headers)
        response.raise_for_status()

        if stream:
            text_parts = []
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    line = line[6:]
                if line and line != "[DONE]":
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        text_parts.append(delta.get("content", ""))
                    except (json.JSONDecodeError, KeyError):
                        continue
            return {"response": "".join(text_parts)}

        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
```

### 4. 带退避的重试

```python
import random

async def with_retry(coro, max_retries: int = 3, base_delay: float = 1.0,
                     max_delay: float = 60.0, retryable_statuses=(429, 502, 503, 504)):
    for attempt in range(max_retries):
        try:
            return await coro
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retryable_statuses:
                raise
            if attempt == max_retries - 1:
                raise
            retry_after = e.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(base_delay * (2 ** attempt) + random.random(), max_delay)
            await asyncio.sleep(delay)
```

### 5. 向后兼容的同步包装器

```python
class OllamaClient:
    def __init__(self, *args, **kwargs):
        self._async_client = AsyncOllamaClient(*args, **kwargs)

    def generate(self, prompt, image_path=None, stream=False, **kwargs):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._async_client.generate(prompt, image_path, stream, **kwargs), loop
            )
            return future.result()
        except RuntimeError:
            return asyncio.run(self._async_client.generate(prompt, image_path, stream, **kwargs))
```

## 实现要点

- 安装依赖：`pip install httpx`
- 更新 `analyzer.py` 使用异步 `generate()`
- 更新 `cli.py` 使用 `asyncio.run()` 运行异步流水线
- 使用 `async with` 确保资源正确释放

## 验证清单

- [ ] 单帧分析与同步版本行为一致
- [ ] 并发请求比串行更快完成
- [ ] 连接池限制被正确遵守
- [ ] 429 速率限制触发退避重试
- [ ] 退出时资源正确释放

## 示例用法

```
/async-llm-client client=both max_connections=10
/async-llm-client client=ollama timeout=120 preserve_sync=true
/async-llm-client client=openai max_connections=20
```
