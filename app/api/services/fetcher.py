"""
Audio Fetcher — 可插拔的音频获取模块

主配置文件 config.yaml 只需指定 `fetcher.type`，各 fetcher 实现的
具体配置参数由子类自身维护（硬编码默认值），不混入主配置文件中。

注册表模式：新增后端只需继承 AudioFetcher 并注册即可，无需改主流程。

Supported types:
  - local_file : 从本地文件系统 / NAS 挂载目录读取
  - s3         : 从 S3 兼容对象存储读取
  - redis      : 从 Redis 读取二进制 blob
  - mysql      : 从 MySQL 读取二进制 blob
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetcher exception
# ---------------------------------------------------------------------------
class FetchError(Exception):
    """Raised when audio cannot be fetched by ID."""
    def __init__(self, message: str, code: str = "FETCH_ERROR") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class AudioFetcher(ABC):
    """
    Abstract base class for all audio fetchers.

    Subclasses must implement fetch() and register themselves via the
    ``register()`` decorator.
    """

    _REGISTRY: ClassVar[Dict[str, Type[AudioFetcher]]] = {}

    @abstractmethod
    def fetch(self, audio_id: str, **kwargs: Any) -> bytes:
        """
        Fetch raw audio bytes by ID.

        Args:
            audio_id: Unique audio identifier.
            **kwargs: Per-request overrides (e.g., bucket name for S3).

        Returns:
            Raw audio bytes (e.g., WAV / ulaw / MP3 file content).

        Raises:
            FetchError: If audio cannot be found or retrieved.
        """
        ...

    # ------------------------------------------------------------------
    # Class methods (registry / factory)
    # ------------------------------------------------------------------

    @classmethod
    def type_name(cls) -> str:
        """Return the type name used in config (default: lowercase class name)."""
        return cls.__name__.lower()

    @classmethod
    def register(cls, type_name: Optional[str] = None) -> Any:
        """Register a fetcher class (used as decorator)."""
        def _decorator(klass: Type[AudioFetcher]) -> Type[AudioFetcher]:
            name = type_name or klass.type_name()
            if name in cls._REGISTRY:
                logger.warning("Overwriting registered fetcher '%s'", name)
            cls._REGISTRY[name] = klass
            logger.debug("Registered fetcher: %s -> %s", name, klass.__name__)
            return klass
        return _decorator

    @classmethod
    def create(cls, type_name: str) -> AudioFetcher:
        """
        Factory: create a fetcher instance by type name.

        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        IMPORTANT: 主配置文件只指定 type，各子类的具体配置由子类自身维护。
        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        Args:
            type_name: Value of ``fetcher.type`` in config (e.g., ``local_file``).

        Returns:
            An initialized AudioFetcher instance with built-in defaults.

        Raises:
            ValueError: If the fetcher type is not registered.
        """
        if type_name not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unknown fetcher type '{type_name}'. "
                f"Available: [{available}]"
            )
        fetcher_cls = cls._REGISTRY[type_name]
        logger.info("Creating fetcher: %s (%s)", type_name, fetcher_cls.__name__)
        return fetcher_cls()


# ===================================================================
# Built-in implementations
# ===================================================================
#
# 每个子类在 __init__ 中维护自己的默认配置，不从外部 config dict 读取。
# 如需自定义行为，直接修改子类默认值或在子类中增加环境变量/专用配置支持。
# ===================================================================


@AudioFetcher.register("local_file")
class LocalFileFetcher(AudioFetcher):
    """
    Fetch audio from local filesystem or NAS mount.

    默认配置在本子类硬编码，如需修改请直接编辑 __init__ 中的默认值，
    或通过继承覆写。

    Defaults:
        base_path:      /mnt/recordings
        extensions:     .wav, .ulaw, .alaw, .mp3, .flac
        search_subdirs: True
    """

    def __init__(self) -> None:
        self._base_path = Path("/mnt/recordings")
        self._extensions: List[str] = [".wav", ".ulaw", ".alaw", ".mp3", ".flac"]
        self._search_subdirs = True

    def fetch(self, audio_id: str, **kwargs: Any) -> bytes:
        # Try direct file first
        for ext in self._extensions:
            candidate = self._base_path / f"{audio_id}{ext}"
            if candidate.exists():
                return candidate.read_bytes()

        # Recursive subdirectory search
        if self._search_subdirs:
            for ext in self._extensions:
                pattern = f"**/{audio_id}{ext}"
                for match in sorted(self._base_path.glob(pattern)):
                    return match.read_bytes()

        raise FetchError(
            f"Audio '{audio_id}' not found in {self._base_path} "
            f"(tried: {', '.join(self._extensions)})"
        )


@AudioFetcher.register("s3")
class S3Fetcher(AudioFetcher):
    """
    Fetch audio from S3-compatible object storage.

    默认配置在本子类硬编码，如需修改请直接编辑 __init__ 中的默认值。
    优先使用环境变量 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY。

    Defaults:
        region:  us-east-1
        bucket:  (empty — must be provided via per-request **kwargs)
    """

    def __init__(self) -> None:
        self._region = "us-east-1"
        self._client = None
        # 注意：bucket 可通过 kwargs 传入，不在 __init__ 中硬编码

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError:
            raise FetchError("boto3 is not installed; run: pip install boto3")
        self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def fetch(self, audio_id: str, **kwargs: Any) -> bytes:
        bucket = kwargs.get("bucket", "")
        if not bucket:
            raise FetchError("S3 bucket is required (pass bucket= in kwargs)")
        s3 = self._get_client()
        try:
            response = s3.get_object(Bucket=bucket, Key=audio_id)
            return response["Body"].read()
        except Exception as e:
            raise FetchError(
                f"S3 fetch failed for '{audio_id}' in bucket '{bucket}': {e}"
            )


@AudioFetcher.register("redis")
class RedisFetcher(AudioFetcher):
    """
    Fetch audio from Redis (stored as binary blob keyed by audio_id).

    配置直接在 __init__ 中硬编码，修改连接参数请直接编辑源码。
    """

    def __init__(self) -> None:
        self._url = "redis://localhost:6379/0"
        self._key_prefix = "audio:"
        self._socket_timeout = 5.0
        self._socket_connect_timeout = 3.0
        self._max_connections = 10
        self._ssl_cert_reqs = "none"

        self._pool: Any = None
        self._redis: Any = None
        self._connect_attempts = 3  # retry count

    # ────────────────────────────────────────────────────────────────────
    # Connection management with retry + connection pool
    # ────────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis as r
        except ImportError:
            raise FetchError("redis-py is not installed; run: pip install redis")

        ssl_kw: Dict[str, Any] = {}
        if self._url.startswith("rediss://") or "ssl=true" in self._url.lower():
            ssl_kw["ssl_cert_reqs"] = self._ssl_cert_reqs

        # Build connection pool
        self._pool = r.ConnectionPool.from_url(
            self._url,
            decode_responses=False,
            socket_timeout=self._socket_timeout,
            socket_connect_timeout=self._socket_connect_timeout,
            max_connections=self._max_connections,
            retry_on_timeout=True,
            health_check_interval=30,
            **ssl_kw,
        )
        self._redis = r.Redis(connection_pool=self._pool)

        # Verify connectivity immediately
        try:
            self._redis.ping()
        except Exception as e:
            self._pool.disconnect()
            self._pool = None
            self._redis = None
            raise FetchError(
                f"Redis ping failed at {self._url}: {e}",
                code="REDIS_CONNECT_FAILED",
            )

        logger.info(
            "RedisFetcher connected to %s (prefix=%r, pool=%d)",
            self._url,
            self._key_prefix,
            self._max_connections,
        )
        return self._redis

    # ────────────────────────────────────────────────────────────────────
    # Fetch with retry
    # ────────────────────────────────────────────────────────────────────

    def fetch(self, audio_id: str, **kwargs: Any) -> bytes:
        key = f"{self._key_prefix}{audio_id}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._connect_attempts + 1):
            try:
                client = self._get_client()
                data = client.get(key)
                if data is None:
                    raise FetchError(
                        f"Audio '{audio_id}' not found in Redis "
                        f"at key '{key}'",
                        code="REDIS_KEY_NOT_FOUND",
                    )
                return bytes(data)
            except FetchError:
                raise  # re-raise key-not-found immediately (no retry)
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Redis fetch attempt %d/%d failed for key=%s: %s",
                    attempt,
                    self._connect_attempts,
                    key,
                    e,
                )
                # Reset connection on failure (trigger reconnect)
                self._redis = None
                if self._pool is not None:
                    try:
                        self._pool.disconnect()
                    except Exception:
                        pass
                    self._pool = None
                if attempt < self._connect_attempts:
                    import time as _time

                    _time.sleep(0.5 * attempt)  # linear backoff

        raise FetchError(
            f"Redis fetch failed for '{audio_id}' "
            f"after {self._connect_attempts} attempts: {last_exc}",
            code="REDIS_FETCH_FAILED",
        )

    # ────────────────────────────────────────────────────────────────────
    # Health check utility
    # ────────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Check Redis connection health. Returns True if connected."""
        try:
            client = self._get_client()
            return client.ping()
        except Exception:
            return False


@AudioFetcher.register("mysql")
class MySQLFetcher(AudioFetcher):
    """
    Fetch audio from MySQL (stored as binary blob in a table).

    配置直接在 __init__ 中硬编码，修改连接参数请直接编辑源码。
    """

    def __init__(self) -> None:
        self._host = "localhost"
        self._port = 3306
        self._user = "root"
        self._password = ""
        self._database = "asv_audio"
        self._table = "audio_store"
        self._id_col = "audio_id"
        self._data_col = "audio_data"
        self._connect_timeout = 5
        self._max_connections = 5
        self._charset = "utf8mb4"

        self._pool: Any = None
        self._connect_attempts = 3

    # ────────────────────────────────────────────────────────────────────
    # Connection pool management
    # ────────────────────────────────────────────────────────────────────

    def _get_connection(self):
        """Get a connection from the pool (lazy init)."""
        if self._pool is not None:
            try:
                conn = self._pool.get_connection()
                # Quick liveness check
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return conn
            except Exception:
                # Connection stale — reset pool
                self._pool = None

        try:
            import pymysql
            from dbutils.pooled_db import PooledDB
        except ImportError:
            raise FetchError(
                "pymysql + DBUtils are required; run: pip install pymysql DBUtils"
            )

        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=self._max_connections,
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            connect_timeout=self._connect_timeout,
            blocking=False,
        )
        conn = self._pool.connection()
        logger.info(
            "MySQLFetcher connected to %s:%d/%s (table=%s, pool=%d)",
            self._host,
            self._port,
            self._database,
            self._table,
            self._max_connections,
        )
        return conn

    # ────────────────────────────────────────────────────────────────────
    # Fetch with retry
    # ────────────────────────────────────────────────────────────────────

    def fetch(self, audio_id: str, **kwargs: Any) -> bytes:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._connect_attempts + 1):
            conn = None
            try:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    sql = (
                        f"SELECT `{self._data_col}` "
                        f"FROM `{self._table}` "
                        f"WHERE `{self._id_col}` = %s"
                    )
                    cur.execute(sql, (audio_id,))
                    row = cur.fetchone()
                if row is None:
                    raise FetchError(
                        f"Audio '{audio_id}' not found in MySQL "
                        f"({self._host}:{self._port}/{self._database}.{self._table})",
                        code="MYSQL_KEY_NOT_FOUND",
                    )
                data = row[0]
                if isinstance(data, bytes):
                    return data
                # pymysql may return a memoryview or other type
                return bytes(data)

            except FetchError:
                raise  # re-raise key-not-found immediately
            except Exception as e:
                last_exc = e
                logger.warning(
                    "MySQL fetch attempt %d/%d failed for id=%s: %s",
                    attempt,
                    self._connect_attempts,
                    audio_id,
                    e,
                )
                # Force pool reconnect on next attempt
                if self._pool is not None:
                    try:
                        self._pool.close()
                    except Exception:
                        pass
                    self._pool = None
                if attempt < self._connect_attempts:
                    import time as _time
                    _time.sleep(0.5 * attempt)

        raise FetchError(
            f"MySQL fetch failed for '{audio_id}' "
            f"after {self._connect_attempts} attempts: {last_exc}",
            code="MYSQL_FETCH_FAILED",
        )

    # ────────────────────────────────────────────────────────────────────
    # Health check utility
    # ────────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Check MySQL connection health. Returns True if connected."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False
