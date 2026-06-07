"""
Configuration management for ASV API service.

Supports:
- YAML config file loading
- Environment variable overrides (prefixed with ASV_)
- Hot-reload of model config
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent

_CONF_DIR = _PKG_DIR / "conf"
_MODEL_DIR = _PKG_DIR / "models"

_CONF_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_PATH_ENV = "ASV_CONFIG_PATH"
_DEFAULT_CONFIG_PATHS = [
    _CONF_DIR / "config.yaml",
    _CONF_DIR / "config.yml",
    Path("./config.yaml"),
    Path("./config.yml"),
    Path.home() / ".asv-api" / "config.yaml",
    Path("/etc/asv-api/config.yaml"),
]

_DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "log_level": "info",
        "max_request_size_mb": 50,
    },
    "model": {
        "path": str(_MODEL_DIR / "campplus.onnx"),
        "provider": "CPUExecutionProvider",
        "provider_options": {},
        "inter_op_threads": 4,
        "intra_op_threads": 4,
        "hot_reload_interval_sec": 30,
    },
    "audio": {
        "target_sample_rate": 16000,
        "min_duration_sec": 0.5,
        "max_duration_sec": 60.0,
        "vad_enabled": True,
        "vad_window_ms": 30,
        "vad_threshold": 0.5,
        "fbank_num_filters": 80,
        "fbank_window_ms": 25,
        "fbank_hop_ms": 10,
    },
    "fetcher": {
        "type": "local_file",
    },
    "verification": {
        "default_threshold": 0.6,
        "scoring_method": "cosine",  # cosine | euclidean | svm
        "normalize_embeddings": True,
    },
    "cache": {
        "enabled": False,
        "backend": "redis",
        "redis_url": "redis://localhost:6379/0",
        "ttl_sec": 86400,
        "max_entries": 100000,
    },
    "storage": {
        "default_backend": "nas",
        "nas_mount_path": "/mnt/recordings",
        "s3_endpoint": "",
        "s3_bucket": "",
        "s3_access_key": "",
        "s3_secret_key": "",
    },
    "thresholds": {
        "customer_service": 0.45,
        "debt_collection": 0.70,
        "audit": None,
    },
    "metrics": {
        "enabled": True,
        "prometheus_port": 8001,
    },
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    max_request_size_mb: int = 50


@dataclass
class ModelConfig:
    path: str = str(_MODEL_DIR / "campplus.onnx")
    provider: str = "CPUExecutionProvider"
    provider_options: Dict[str, Any] = field(default_factory=dict)
    inter_op_threads: int = 4
    intra_op_threads: int = 4
    hot_reload_interval_sec: int = 30


@dataclass
class AudioConfig:
    target_sample_rate: int = 16000
    min_duration_sec: float = 0.5
    max_duration_sec: float = 60.0
    vad_enabled: bool = True
    vad_window_ms: int = 30
    vad_threshold: float = 0.5
    fbank_num_filters: int = 80
    fbank_window_ms: int = 25
    fbank_hop_ms: int = 10


@dataclass
class VerificationConfig:
    default_threshold: float = 0.6
    scoring_method: str = "cosine"
    normalize_embeddings: bool = True


@dataclass
class CacheConfig:
    enabled: bool = False
    backend: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    ttl_sec: int = 86400
    max_entries: int = 100000


@dataclass
class StorageConfig:
    default_backend: str = "nas"
    nas_mount_path: str = "/mnt/recordings"
    s3_endpoint: str = ""
    s3_bucket: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    download_dir: str = ""  # empty => defaults to <api>/downloads/


@dataclass
class ThresholdsConfig:
    customer_service: Optional[float] = 0.45
    debt_collection: Optional[float] = 0.70
    audit: Optional[float] = None


@dataclass
class MetricsConfig:
    enabled: bool = True
    prometheus_port: int = 8001


@dataclass
class TrainingConfig:
    recordings_root: str = ""
    preprocessed_root: str = ""
    db_path: str = ""

    def resolve(self, pkg_dir: Path) -> None:
        """Resolve empty paths to defaults under <pkg_dir>/../data/."""
        data_root = pkg_dir.parent / "data"
        if not self.recordings_root:
            self.recordings_root = str(data_root / "local_recordings")
        if not self.preprocessed_root:
            self.preprocessed_root = str(data_root / "preprocessed")
        if not self.db_path:
            self.db_path = str(data_root / "training.db")


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fetcher_type: str = "local_file"
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    _model_md5: Optional[str] = None

    @property
    def model_path_hash(self) -> str:
        """MD5 hash of the model file path for change detection."""
        if self._model_md5 is None:
            self._model_md5 = self._compute_md5()
        return self._model_md5

    def _compute_md5(self) -> str:
        """Compute MD5 of the model config path."""
        raw = f"{self.model.path}::{self.model.provider}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def invalidate_model_hash(self) -> None:
        self._model_md5 = None

    def get_scenario_threshold(self, scenario: str) -> Optional[float]:
        """Get threshold for a named scenario."""
        mapping = {
            "customer_service": self.thresholds.customer_service,
            "debt_collection": self.thresholds.debt_collection,
            "audit": self.thresholds.audit,
        }
        return mapping.get(scenario)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env_override(prefix: str, key: str, value: Any) -> Any:
    """Override a config value from environment variable if set."""
    env_key = f"{prefix}{key.upper().replace('.', '_')}"
    env_val = os.environ.get(env_key)
    if env_val is None:
        return value
    # Try to cast to the original type
    if isinstance(value, bool):
        return env_val.lower() in ("1", "true", "yes")
    if isinstance(value, int):
        return int(env_val)
    if isinstance(value, float):
        return float(env_val)
    return env_val


def _deep_merge(base: Dict, overrides: Dict, prefix: str = "ASV_") -> Dict:
    """Deep merge overrides into base with env var support."""
    result = dict(base)
    for key, val in overrides.items():
        full_key = f"{prefix}{key.upper()}" if prefix == "ASV_" else f"{prefix}.{key}"
        if isinstance(val, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], val, full_key)
        else:
            result[key] = _env_override(prefix, f"{prefix}.{key}" if prefix != "ASV_" else key, val)
    return result


def _find_config_file() -> Optional[Path]:
    """Find the first existing config file from default paths or env var."""
    env_path = os.environ.get(_CONFIG_PATH_ENV)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config_from_dict(data: Dict[str, Any]) -> AppConfig:
    """Load config from a dictionary (YAML-parsed)."""
    cfg = AppConfig()

    # Server
    if "server" in data:
        for k, v in data["server"].items():
            if hasattr(cfg.server, k):
                setattr(cfg.server, k, v)

    # Model — resolve relative path against PKG_DIR
    if "model" in data:
        for k, v in data["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
        # Resolve relative model path against the api/ package directory
        raw_path = cfg.model.path
        if raw_path and not Path(raw_path).is_absolute():
            resolved = _PKG_DIR / raw_path
            cfg.model.path = str(resolved)

    # Audio
    if "audio" in data:
        for k, v in data["audio"].items():
            if hasattr(cfg.audio, k):
                setattr(cfg.audio, k, v)

    # Fetcher  — 只读取 type 字段，其余交由子类自身维护
    if "fetcher" in data and isinstance(data["fetcher"], dict):
        f_type = data["fetcher"].get("type")
        if f_type:
            cfg.fetcher_type = str(f_type)

    # Verification
    if "verification" in data:
        for k, v in data["verification"].items():
            if hasattr(cfg.verification, k):
                setattr(cfg.verification, k, v)

    # Cache
    if "cache" in data:
        for k, v in data["cache"].items():
            if hasattr(cfg.cache, k):
                setattr(cfg.cache, k, v)

    # Storage
    if "storage" in data:
        for k, v in data["storage"].items():
            if hasattr(cfg.storage, k):
                setattr(cfg.storage, k, v)

    # Thresholds
    if "thresholds" in data:
        for k, v in data["thresholds"].items():
            if hasattr(cfg.thresholds, k):
                setattr(cfg.thresholds, k, v)

    # Metrics
    if "metrics" in data:
        for k, v in data["metrics"].items():
            if hasattr(cfg.metrics, k):
                setattr(cfg.metrics, k, v)

    # Training (recording storage)
    if "training" in data:
        for k, v in data["training"].items():
            if hasattr(cfg.training, k) and not isinstance(v, dict):
                setattr(cfg.training, k, v)

    # Resolve training paths
    cfg.training.resolve(_PKG_DIR)

    return cfg


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Override precedence (highest wins):
      1. Environment variables (ASV_SERVER_PORT, ASV_MODEL_PATH, ...)
      2. YAML config file
      3. Built-in defaults
    """
    if config_path is None:
        config_path = _find_config_file()

    raw: Dict[str, Any] = dict(_DEFAULT_CONFIG)

    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, file_cfg)

    # Apply env overrides (flat key lookup)
    for section, values in raw.items():
        for key in values:
            env_key = f"ASV_{section.upper()}_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                orig = raw[section][key]
                if isinstance(orig, bool):
                    raw[section][key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(orig, int):
                    raw[section][key] = int(env_val)
                elif isinstance(orig, float):
                    raw[section][key] = float(env_val)
                else:
                    raw[section][key] = env_val

    return load_config_from_dict(raw)


def dump_config(cfg: AppConfig) -> Dict[str, Any]:
    """Serialize AppConfig to a dict for logging/debugging (redact secrets)."""
    data = {
        "server": {
            "host": cfg.server.host,
            "port": cfg.server.port,
            "workers": cfg.server.workers,
            "log_level": cfg.server.log_level,
            "max_request_size_mb": cfg.server.max_request_size_mb,
        },
        "model": {
            "path": cfg.model.path,
            "provider": cfg.model.provider,
            "hot_reload_interval_sec": cfg.model.hot_reload_interval_sec,
        },
        "audio": {
            "target_sample_rate": cfg.audio.target_sample_rate,
            "min_duration_sec": cfg.audio.min_duration_sec,
            "max_duration_sec": cfg.audio.max_duration_sec,
            "vad_enabled": cfg.audio.vad_enabled,
        },
        "fetcher": {
            "type": cfg.fetcher_type,
        },
        "verification": {
            "default_threshold": cfg.verification.default_threshold,
            "scoring_method": cfg.verification.scoring_method,
        },
        "cache": {
            "enabled": cfg.cache.enabled,
            "backend": cfg.cache.backend,
            "redis_url": "***redacted***",
            "ttl_sec": cfg.cache.ttl_sec,
        },
        "storage": {
            "default_backend": cfg.storage.default_backend,
            "nas_mount_path": cfg.storage.nas_mount_path,
            "s3_endpoint": cfg.storage.s3_endpoint or "",
            "download_dir": cfg.storage.download_dir or "",
        },
        "thresholds": {
            "customer_service": cfg.thresholds.customer_service,
            "debt_collection": cfg.thresholds.debt_collection,
            "audit": cfg.thresholds.audit,
        },
    }
    return data


# Module-level singleton: load config once at import time
app_config: AppConfig = load_config()
