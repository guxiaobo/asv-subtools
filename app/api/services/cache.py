"""
Embedding cache service.

Stores pre-computed speaker embeddings for high-frequency speakers,
reducing inference latency from ~50ms (model inference) to ~1ms (lookup).

Supports:
- Redis backend (production)
- In-memory LRU backend (fallback, no Redis dependency)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple
from collections import OrderedDict

import numpy as np

from config import CacheConfig

logger = logging.getLogger(__name__)


class CacheMiss(Exception):
    """Raised when an embedding is not found in cache."""
    pass


class EmbeddingCache:
    """
    Abstract cache interface for speaker embeddings.

    Usage::

        cache = create_cache(config)
        cache.set("speaker:123", embedding)
        emb, meta = cache.get("speaker:123")  # -> ndarray, metadata_dict
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._enabled = config.enabled
        self._ttl_sec = config.ttl_sec
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set(self, key: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Store an embedding."""
        raise NotImplementedError

    def get(self, key: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """Retrieve an embedding. Raises CacheMiss if not found."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove an embedding from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cached embeddings."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""
        pass


class MemoryEmbeddingCache(EmbeddingCache):
    """
    In-memory LRU cache for embeddings.

    Uses OrderedDict for O(1) get/set with LRU eviction.
    Default max entries: 100,000 (≈ 100K × 512 × 4 bytes = ~200MB for float32).
    """

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)
        self._max_entries = config.max_entries
        self._store: OrderedDict = OrderedDict()
        logger.info(
            "Memory cache initialized (max_entries=%d, ttl=%ds)",
            self._max_entries, self._ttl_sec,
        )

    def set(self, key: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if not self._enabled:
            return

        # LRU eviction
        if len(self._store) >= self._max_entries:
            self._store.popitem(last=False)

        self._store[key] = {
            "embedding": embedding.copy(),
            "metadata": metadata or {},
            "created_at": time.time(),
        }

    def get(self, key: str) -> np.ndarray:
        if not self._enabled:
            raise CacheMiss("Cache disabled")

        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            raise CacheMiss(f"Key '{key}' not found")

        # TTL check
        age = time.time() - entry["created_at"]
        if age > self._ttl_sec:
            self._store.pop(key, None)
            self._misses += 1
            raise CacheMiss(f"Key '{key}' expired (age={age:.1f}s)")

        # LRU refresh: move to end
        self._store.move_to_end(key)
        self._hits += 1
        return entry["embedding"]

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


class RedisEmbeddingCache(EmbeddingCache):
    """
    Redis-backed embedding cache for production use.

    Embedding stored as binary: 4 bytes for dimension, rest for float32 data.
    """

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)
        self._redis_url = config.redis_url
        self._client = self._connect()

    def _connect(self):
        try:
            import redis as redis_module
            client = redis_module.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()
            logger.info("Redis cache connected: %s", self._redis_url)
            return client
        except ImportError:
            logger.warning("redis module not installed, cache disabled")
            self._enabled = False
            return None
        except Exception as e:
            logger.warning("Redis connection failed: %s, cache disabled", e)
            self._enabled = False
            return None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._enabled

    def set(self, key: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if not self.connected:
            return

        # Serialize: [dim:4B][float32_data...]
        arr = np.ascontiguousarray(embedding, dtype=np.float32)
        payload = arr.tobytes()
        redis_key = f"embed:{key}"

        try:
            self._client.setex(redis_key, self._ttl_sec, payload)
        except Exception as e:
            logger.error("Redis set failed for '%s': %s", key, e)

    def get(self, key: str) -> np.ndarray:
        if not self.connected:
            self._misses += 1
            raise CacheMiss("Redis cache not connected")

        redis_key = f"embed:{key}"
        try:
            data = self._client.get(redis_key)
        except Exception as e:
            logger.error("Redis get failed for '%s': %s", key, e)
            self._misses += 1
            raise CacheMiss("Redis error")

        if data is None:
            self._misses += 1
            raise CacheMiss(f"Key '{redis_key}' not found")

        embedding = np.frombuffer(data, dtype=np.float32).copy()
        self._hits += 1
        return embedding

    def delete(self, key: str) -> None:
        if self.connected:
            try:
                self._client.delete(f"embed:{key}")
            except Exception as e:
                logger.error("Redis delete failed for '%s': %s", key, e)

    def clear(self) -> None:
        if self.connected:
            try:
                for key in self._client.scan_iter(match="embed:*"):
                    self._client.delete(key)
            except Exception as e:
                logger.error("Redis clear failed: %s", e)

    def close(self) -> None:
        if self._client:
            self._client.close()


def create_cache(config: CacheConfig) -> EmbeddingCache:
    """
    Factory: create an EmbeddingCache instance based on config.

    Falls back to MemoryEmbeddingCache if Redis is unavailable or disabled.
    """
    if not config.enabled:
        return MemoryEmbeddingCache(config)  # no-op since enabled=False

    if config.backend == "redis":
        try:
            cache = RedisEmbeddingCache(config)
            if cache.connected:
                return cache
            logger.warning("Redis unavailable, falling back to memory cache")
        except Exception as e:
            logger.warning("Redis init failed (%s), falling back to memory cache", e)

    return MemoryEmbeddingCache(config)
