"""
训练模块配置管理。

从 YAML 配置文件加载训练相关参数，
与 API 共享配置结构（兼容 config.py 的 AppConfig）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "db_path": "",  # empty => <proj_root>/data/training.db
    "recordings_root": "/data/local_recordings",
    "preprocessed_root": "/data/preprocessed",
    "test_set_path": "data/test_set",
    "preprocessing": {
        "target_sample_rate": 16000,
        "min_segment_sec": 0.5,
        "max_segment_sec": 15.0,
        "snr_threshold": 4.0,
        "vad_window_ms": 30,
        "vad_threshold": 0.5,
        "filter_leading_sec": 2.0,
    },
    "incremental_train": {
        "base_lr": 0.0001,
        "epochs": 3,
        "batch_size": 64,
        "improvement_threshold": 0.001,
    },
    "model": {
        "api_models_dir": "",  # empty => api/models/
        "checkpoint_path": "",
        "backbone": "CAM++",
        "embedding_dim": 192,
    },
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def find_config_file() -> Optional[Path]:
    """查找训练配置文件。搜索优先级：环境变量 > 默认路径。"""
    env_path = os.environ.get("ASV_TRAIN_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path("./train_config.yaml"),
        Path("./train_config.yml"),
        Path("./conf/train_config.yaml"),
        Path(__file__).resolve().parent / "conf" / "train_config.yaml",
        Path(__file__).resolve().parent.parent / "api" / "conf" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载训练配置。

    如果未指定 config_path，自动搜索。找不到则返回默认值。
    """
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)

    if config_path is None:
        config_path = find_config_file()

    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        # Merge top-level training section if exists
        training_cfg = file_cfg.get("training", file_cfg)
        for key, val in training_cfg.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val

    # Resolve relative paths
    config = _resolve_paths(config)

    return config


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """将相对路径解析为相对于项目根的绝对路径。"""
    proj_root = Path(__file__).resolve().parent.parent  # app/

    if config.get("db_path"):
        p = Path(config["db_path"])
        if not p.is_absolute():
            config["db_path"] = str(proj_root / p)
    else:
        config["db_path"] = str(proj_root / "data" / "training.db")

    if config.get("preprocessed_root"):
        p = Path(config["preprocessed_root"])
        if not p.is_absolute():
            config["preprocessed_root"] = str(proj_root / p)
    else:
        config["preprocessed_root"] = str(proj_root / "data" / "preprocessed")

    if config.get("recordings_root"):
        p = Path(config["recordings_root"])
        if not p.is_absolute():
            config["recordings_root"] = str(proj_root / p)
    else:
        config["recordings_root"] = str(proj_root / "data" / "local_recordings")

    if config.get("test_set_path"):
        p = Path(config["test_set_path"])
        if not p.is_absolute():
            config["test_set_path"] = str(proj_root / p)
    else:
        config["test_set_path"] = str(proj_root / "data" / "test_set")

    # Model: api/models/ directory
    models_dir = config.get("model", {}).get("api_models_dir", "")
    if models_dir:
        p = Path(models_dir)
        if not p.is_absolute():
            config["model"]["api_models_dir"] = str(proj_root.parent / p)
    else:
        config["model"]["api_models_dir"] = str(proj_root / "api" / "models")

    return config
