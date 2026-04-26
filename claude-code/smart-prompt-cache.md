# smart-prompt-cache

## 概述

为 video-analyzer 添加多级缓存机制，消除冗余 I/O 和重复 LLM 推理。包含提示词模板缓存（内存 LRU）、分析结果缓存（SQLite 磁盘缓存）和配置缓存（文件变更感知）。

## 适用场景

- 同一视频多次运行分析
- 调优提示词后反复测试
- 处理包含相同片段的多视频（如片头片尾）
- 提示词文件在运行期间不变

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| cache_dir | string | "~/.cache/video-analyzer" | 缓存目录 |
| max_cache_mb | float | 1024 | 缓存大小上限（MB） |
| cache_ttl_hours | float | 168 | 缓存生存时间（小时，默认 7 天） |
| cache_analyses | bool | true | 是否缓存 LLM 分析结果 |

## 核心指令

实现三级缓存系统：

### 1. 提示词加载器缓存

替换 `video_analyzer/prompt.py`：

```python
import functools
import hashlib
from pathlib import Path
from typing import Optional
import importlib.resources

class CachedPromptLoader:
    def __init__(self, prompt_dir, prompts, cache_dir: Optional[str] = None):
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.prompts = prompts or []
        self._cache = {}
        self._path_index = {}
        self._build_path_index()

    def _build_path_index(self):
        try:
            pkg_path = Path(importlib.resources.files('video_analyzer')) / 'prompts'
            if pkg_path.exists():
                for f in pkg_path.glob("*.txt"):
                    self._path_index[f.stem] = f
        except ImportError:
            pass
        if self.prompt_dir and self.prompt_dir.exists():
            for f in self.prompt_dir.glob("*.txt"):
                self._path_index[f.stem] = f

    @functools.lru_cache(maxsize=32)
    def get_by_index(self, index: int) -> str:
        if index >= len(self.prompts):
            raise IndexError(f"Prompt index {index} out of range")
        return self._load_prompt(self.prompts[index])

    @functools.lru_cache(maxsize=32)
    def get_by_name(self, name: str) -> str:
        path = self._path_index.get(name)
        if not path:
            raise FileNotFoundError(f"Prompt '{name}' not found")
        return self._load_prompt(str(path))

    def _load_prompt(self, prompt_path: str) -> str:
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return path.read_text().strip()

    def invalidate_cache(self):
        self.get_by_index.cache_clear()
        self.get_by_name.cache_clear()
        self._cache.clear()
```

### 2. 磁盘分析结果缓存（SQLite）

创建 `video_analyzer/cache.py`：

```python
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional
import sqlite3

class AnalysisCache:
    def __init__(self, cache_dir: str = "~/.cache/video-analyzer",
                 max_size_mb: float = 1024.0, ttl_hours: float = 168.0):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_hours * 3600
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, value BLOB, created_at REAL, size_bytes INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")
            conn.commit()

    def _compute_key(self, prompt: str, image_path: str, model: str, temperature: float = 0.7) -> str:
        img_mtime = Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        content = f"{prompt}:{image_path}:{img_mtime}:{model}:{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, image_path: str, model: str, temperature: float = 0.7) -> Optional[Any]:
        key = self._compute_key(prompt, image_path, model, temperature)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value, created_at FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            value, created_at = row
            if time.time() - created_at > self.ttl_seconds:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return pickle.loads(value)

    def set(self, prompt: str, image_path: str, model: str, temperature: float, value: Any):
        key = self._compute_key(prompt, image_path, model, temperature)
        serialized = pickle.dumps(value)
        size = len(serialized)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, size_bytes) VALUES (?, ?, ?, ?)",
                (key, serialized, time.time(), size)
            )
            conn.commit()
        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache").fetchone()[0]
            if total < self.max_size_mb * 1024 * 1024:
                return
            to_delete = []
            current_size = total
            for key, size in conn.execute("SELECT key, size_bytes FROM cache ORDER BY created_at"):
                if current_size < self.max_size_mb * 1024 * 1024 * 0.8:
                    break
                to_delete.append(key)
                current_size -= size
            for key in to_delete:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            count, total_size = conn.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache").fetchone()
        return {"entries": count, "size_mb": total_size / (1024 * 1024), "max_size_mb": self.max_size_mb}
```

### 3. 带缓存的 LLM 客户端包装器

```python
class CachedLLMClient:
    def __init__(self, client, cache: Optional[AnalysisCache] = None):
        self.client = client
        self.cache = cache

    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> dict:
        if stream or self.cache is None or image_path is None:
            return await self._generate(prompt, image_path, stream, **kwargs)
        model = getattr(self.client, 'model', 'unknown')
        temperature = kwargs.get('temperature', 0.7)
        cached = self.cache.get(prompt, image_path, model, temperature)
        if cached is not None:
            return cached
        result = await self._generate(prompt, image_path, stream, **kwargs)
        self.cache.set(prompt, image_path, model, temperature, result)
        return result

    async def _generate(self, prompt, image_path, stream, **kwargs):
        if hasattr(self.client, 'generate_async'):
            return await self.client.generate_async(prompt, image_path, stream, **kwargs)
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.generate(prompt, image_path, stream, **kwargs)
        )
```

### 4. 配置缓存

```python
import json
from pathlib import Path

class CachedConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._cached_config = None
        self._cached_mtime = 0

    @property
    def config(self) -> dict:
        if not self.config_path.exists():
            return {}
        mtime = self.config_path.stat().st_mtime
        if mtime != self._cached_mtime or self._cached_config is None:
            self._cached_config = json.loads(self.config_path.read_text())
            self._cached_mtime = mtime
        return self._cached_config

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
```

## 实现要点

- 在 `cli.py` 中包装 LLM 客户端：`llm_client = CachedLLMClient(raw_client, cache=cache)`
- 缓存键包含图片文件修改时间，文件变更自动失效缓存
- 提供缓存统计和清空命令

## 验证清单

- [ ] 同一视频第二次运行速度提升 50% 以上
- [ ] 日志中显示缓存命中率
- [ ] 修改提示词文件后相关缓存失效
- [ ] 缓存遵守 TTL 和大小限制
- [ ] SQLite 缓存在并行分析中线程安全

## 示例用法

```
/smart-prompt-cache max_cache_mb=1024 cache_ttl_hours=168
/smart-prompt-cache cache_dir="~/.cache/video-analyzer" cache_analyses=true
```
