# async-llm-client

Convert synchronous LLM clients to async for concurrent frame analysis and better resource utilization.

## Description

Replaces the blocking `requests`-based LLM clients in video-analyzer with async `httpx`-based implementations, enabling concurrent API calls, connection pooling, and non-blocking I/O throughout the pipeline.

## When to use

- When parallel frame analysis is implemented
- When using cloud APIs with rate limits (async + semaphore = controlled concurrency)
- When integrating with async web frameworks (FastAPI, etc.)
- When the UI needs to remain responsive during analysis

## Parameters

- client: "ollama" | "openai" | "both"
- max_connections: Connection pool size (default: 10)
- timeout: Request timeout in seconds (default: 60)
- preserve_sync: Keep sync API as wrapper (default: true)

## Implementation

### 1. Async Base Client

Create `video_analyzer/clients/async_llm_client.py`:

```python
import base64
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import httpx
from functools import lru_cache
import os

class AsyncLLMClient(ABC):
    def __init__(self, timeout: float = 60.0, max_connections: int = 10):
        self.timeout = timeout
        limits = httpx.Limits(max_connections=max_connections)
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)

    async def encode_image(self, image_path: str) -> str:
        """Async image encoding with LRU cache support."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime, stat.st_size)

        # Check cache
        cached = self._get_cached_image(cache_key)
        if cached:
            return cached

        async with asyncio.Lock():
            # Double-check after acquiring lock
            cached = self._get_cached_image(cache_key)
            if cached:
                return cached

            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            self._set_cached_image(cache_key, encoded)
            return encoded

    def _get_cached_image(self, cache_key):
        # Implement LRU cache storage
        pass

    def _set_cached_image(self, cache_key, value):
        pass

    @abstractmethod
    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, image_path: Optional[str] = None,
                              **kwargs) -> AsyncIterator[str]:
        pass

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
```

### 2. Async Ollama Client

```python
import json
import httpx
from typing import Optional, Dict, Any, AsyncIterator

class AsyncOllamaClient(AsyncLLMClient):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2-vision",
                 timeout: float = 120.0, max_connections: int = 10):
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
        else:
            return response.json()

    async def generate_stream(self, prompt: str, image_path: Optional[str] = None,
                              **kwargs) -> AsyncIterator[str]:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": kwargs.get("num_predict", 300)}
        }

        if image_path:
            data["images"] = [await self.encode_image(image_path)]

        async with self.client.stream("POST", self.generate_url, json=data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
```

### 3. Async OpenAI-Compatible Client

```python
import json
from typing import Optional, Dict, Any, AsyncIterator

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
        else:
            result = response.json()
            return {"response": result["choices"][0]["message"]["content"]}

    async def generate_stream(self, prompt: str, image_path: Optional[str] = None,
                              **kwargs) -> AsyncIterator[str]:
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
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 300),
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.client.stream("POST", self.chat_url, json=data, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    line = line[6:]
                if line and line != "[DONE]":
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError):
                        continue
```

### 4. Backward-Compatible Sync Wrappers

```python
class OllamaClient:
    """Sync wrapper around AsyncOllamaClient for backward compatibility."""
    def __init__(self, *args, **kwargs):
        self._async_client = AsyncOllamaClient(*args, **kwargs)
        self._loop = None

    def generate(self, prompt, image_path=None, stream=False, **kwargs):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._async_client.generate(prompt, image_path, stream, **kwargs),
                loop
            )
            return future.result()
        except RuntimeError:
            # No running loop - use asyncio.run
            return asyncio.run(self._async_client.generate(prompt, image_path, stream, **kwargs))

    def close(self):
        import asyncio
        try:
            asyncio.run(self._async_client.close())
        except:
            pass
```

### 5. Retry with Exponential Backoff

```python
import random

async def with_retry(coro, max_retries: int = 3, base_delay: float = 1.0,
                     max_delay: float = 60.0, retryable_statuses=(429, 502, 503, 504)):
    """Execute coroutine with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await coro
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retryable_statuses:
                raise

            if attempt == max_retries - 1:
                raise

            # Check for Retry-After header
            retry_after = e.response.headers.get("Retry-After")
            if retry_after:
                delay = float(retry_after)
            else:
                delay = min(base_delay * (2 ** attempt) + random.random(), max_delay)

            await asyncio.sleep(delay)
```

## Integration Checklist

- [ ] Install `httpx`: `pip install httpx`
- [ ] Create async client classes
- [ ] Update `analyzer.py` to use async `generate()`
- [ ] Update `cli.py` to run async pipeline with `asyncio.run()`
- [ ] Add `async with` context management for proper cleanup
- [ ] Verify sync wrappers preserve existing CLI behavior
- [ ] Test concurrent frame analysis with semaphore

## Dependencies

```
httpx>=0.27.0
```

## Verification

- [ ] Single frame analysis works identically to sync version
- [ ] Concurrent requests complete faster than sequential
- [ ] Connection pool limits are respected
- [ ] Rate limit 429 triggers retry with backoff
- [ ] Resources are properly cleaned up on exit
