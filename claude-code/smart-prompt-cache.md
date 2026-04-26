# smart-prompt-cache

Add intelligent caching to prompt loading, frame analysis results, and configuration for faster repeated runs.

## Description

Eliminates redundant I/O and repeated LLM inference by caching prompt templates, image encodings, and analysis results. Significantly speeds up re-runs, testing, and iterative prompt tuning.

## When to use

- When running analysis multiple times on the same video
- When tuning prompts and re-running the same frames
- When the same frames appear in multiple videos (e.g., intros)
- When prompt files don't change between runs

## Parameters

- cache_dir: Directory for cache storage (default: `~/.cache/video-analyzer`)
- max_cache_mb: Maximum cache size in MB (default: 1024)
- cache_analyses: Whether to cache LLM analysis results (default: true)
- cache_ttl_hours: Cache time-to-live in hours (default: 168 = 7 days)

## Implementation

### 1. Prompt Loader Cache

Replace `video_analyzer/prompt.py`:

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
        self._cache = {}  # In-memory cache
        self._path_index = {}  # name -> path mapping

        # Build path index at init
        self._build_path_index()

    def _build_path_index(self):
        """Pre-scan all prompt locations to avoid repeated disk checks."""
        # Package prompts
        try:
            pkg_path = Path(importlib.resources.files('video_analyzer')) / 'prompts'
            if pkg_path.exists():
                for f in pkg_path.glob("*.txt"):
                    self._path_index[f.stem] = f
        except ImportError:
            pass

        # User prompts
        if self.prompt_dir and self.prompt_dir.exists():
            for f in self.prompt_dir.glob("*.txt"):
                self._path_index[f.stem] = f

    @functools.lru_cache(maxsize=32)
    def get_by_index(self, index: int) -> str:
        """Cached by index. LRU cache avoids repeated file reads."""
        if index >= len(self.prompts):
            raise IndexError(f"Prompt index {index} out of range")
        return self._load_prompt(self.prompts[index])

    @functools.lru_cache(maxsize=32)
    def get_by_name(self, name: str) -> str:
        """Cached by name."""
        path = self._path_index.get(name)
        if not path:
            raise FileNotFoundError(f"Prompt '{name}' not found in {list(self._path_index.keys())}")
        return self._load_prompt(str(path))

    def _load_prompt(self, prompt_path: str) -> str:
        """Load prompt from pre-resolved path."""
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return path.read_text().strip()

    def invalidate_cache(self):
        """Clear all caches. Call when prompt files change."""
        self.get_by_index.cache_clear()
        self.get_by_name.cache_clear()
        self._cache.clear()
```

### 2. Disk-Based Result Cache

Create `video_analyzer/cache.py`:

```python
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional
import sqlite3

class AnalysisCache:
    """Disk-backed cache for LLM analysis results using SQLite."""

    def __init__(self, cache_dir: str = "~/.cache/video-analyzer",
                 max_size_mb: float = 1024.0,
                 ttl_hours: float = 168.0):
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
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
            """)
            conn.commit()

    def _compute_key(self, prompt: str, image_path: str, model: str,
                     temperature: float = 0.7) -> str:
        """Compute deterministic cache key."""
        # Include image modification time for cache invalidation
        img_mtime = Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        content = f"{prompt}:{image_path}:{img_mtime}:{model}:{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, image_path: str, model: str,
            temperature: float = 0.7) -> Optional[Any]:
        """Get cached analysis result if valid."""
        key = self._compute_key(prompt, image_path, model, temperature)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, created_at FROM cache WHERE key = ?",
                (key,)
            ).fetchone()

            if row is None:
                return None

            value, created_at = row
            if time.time() - created_at > self.ttl_seconds:
                # Expired
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None

            return pickle.loads(value)

    def set(self, prompt: str, image_path: str, model: str,
            temperature: float, value: Any):
        """Store analysis result in cache."""
        key = self._compute_key(prompt, image_path, model, temperature)
        serialized = pickle.dumps(value)
        size = len(serialized)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache (key, value, created_at, size_bytes)
                   VALUES (?, ?, ?, ?)""",
                (key, serialized, time.time(), size)
            )
            conn.commit()

        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        """Remove old entries if cache exceeds max size."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM cache"
            ).fetchone()[0]

            if total < self.max_size_mb * 1024 * 1024:
                return

            # Delete oldest entries until under limit
            to_delete = []
            cursor = conn.execute(
                "SELECT key, size_bytes FROM cache ORDER BY created_at"
            )
            current_size = total
            for key, size in cursor:
                if current_size < self.max_size_mb * 1024 * 1024 * 0.8:
                    break
                to_delete.append(key)
                current_size -= size

            for key in to_delete:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        """Clear all cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def stats(self) -> dict:
        """Return cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            count, total_size = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache"
            ).fetchone()

        return {
            "entries": count,
            "size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "db_path": str(self.db_path),
        }
```

### 3. Cached LLM Client Wrapper

```python
class CachedLLMClient:
    """Wrapper that adds caching to any LLM client."""

    def __init__(self, client, cache: Optional[AnalysisCache] = None):
        self.client = client
        self.cache = cache

    async def generate(self, prompt: str, image_path: Optional[str] = None,
                       stream: bool = False, **kwargs) -> dict:
        # Don't cache streaming requests
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
        # Delegate to wrapped client
        if hasattr(self.client, 'generate_async'):
            return await self.client.generate_async(prompt, image_path, stream, **kwargs)
        # Fallback to sync wrapped in thread
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.generate(prompt, image_path, stream, **kwargs)
        )
```

### 4. Configuration Cache

```python
import functools
import json
from pathlib import Path

class CachedConfig:
    """Config with file-change-aware caching."""

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

## Integration

In `cli.py`, wrap the analyzer:
```python
cache = AnalysisCache(
    cache_dir=config.get("cache_dir", "~/.cache/video-analyzer"),
    max_size_mb=config.get("max_cache_mb", 1024),
    ttl_hours=config.get("cache_ttl_hours", 168),
)

# Wrap LLM client with cache
llm_client = CachedLLMClient(raw_client, cache=cache)
```

## Verification

- [ ] Second run of same video is >50% faster
- [ ] Cache hit rate is reported in logs
- [ ] Changing prompt file invalidates relevant cache entries
- [ ] Cache respects TTL and max size limits
- [ ] SQLite cache is thread-safe for parallel analysis
