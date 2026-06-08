"""
模型版本管理模块。

负责 ONNX 模型导出、版本编号、存储和数据库记录。
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

from train.db import (
    activate_model,
    deactivate_model,
    get_active_model,
    get_connection,
    insert_model_version,
)

logger = logging.getLogger("train.model_manager")

# ---------------------------------------------------------------------------
# Version naming
# ---------------------------------------------------------------------------


def get_next_version(conn) -> str:
    """计算下一个模型版本号。"""
    active = get_active_model(conn)
    if active is None:
        return "v1.0"

    current = active["version"]
    try:
        parts = current.lstrip("v").split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return f"v{major}.{minor + 1}"
    except (ValueError, IndexError):
        return "v1.1"


# ---------------------------------------------------------------------------
# MD5
# ---------------------------------------------------------------------------


def compute_md5(file_path: str) -> str:
    """计算文件 MD5。"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(
    model_checkpoint_path: str,
    output_path: str,
    embedding_dim: int = 192,
) -> str:
    """
    将 PyTorch checkpoint 导出为 ONNX 模型。

    这是一个骨架方法，实际导出需要 ASV-Subtools 的 export_onnx.py 调用。

    Args:
        model_checkpoint_path: PyTorch checkpoint (.pt / .pth)。
        output_path: ONNX 输出路径。
        embedding_dim: embedding 维度。

    Returns:
        导出的 ONNX 文件路径。
    """
    logger.info("Exporting ONNX: %s → %s", model_checkpoint_path, output_path)

    # TODO: 接入 ASV-Subtools 的 export_onnx.py
    # pytorch/pipeline/export_onnx.py --checkpoint <ckpt> --output <onnx>

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 模拟导出：如果 checkpoint 存在，生成一个占位
    if Path(model_checkpoint_path).exists():
        # In real implementation, this would call:
        #   subprocess.run([
        #       "python", "pytorch/pipeline/export_onnx.py",
        #       "--checkpoint", model_checkpoint_path,
        #       "--output", str(output),
        #   ], check=True)
        logger.warning("ONNX 导出未真正执行（需要 ASV-Subtools export_onnx.py）")
    else:
        logger.warning("Checkpoint %s 不存在，跳过 ONNX 导出", model_checkpoint_path)

    return str(output)


# ---------------------------------------------------------------------------
# Model publishing
# ---------------------------------------------------------------------------


def publish_model(
    onnx_path: str,
    api_models_dir: str,
    version: str,
) -> str:
    """
    将 ONNX 模型发布到 api/models/ 目录。

    执行步骤：
    1. 复制 ONNX 到 api/models/{version}.onnx
    2. 复制/替换 api/models/campplus.onnx（ASV API 热加载的文件）

    Args:
        onnx_path: 源 ONNX 文件路径。
        api_models_dir: API 模型目录路径。
        version: 模型版本号（如 v1.1）。

    Returns:
        api/models/ 中的目标路径。
    """
    models_dir = Path(api_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy as versioned file
    versioned_path = models_dir / f"{version}.onnx"
    src_path = Path(onnx_path).resolve()
    dst_path = versioned_path.resolve()
    if src_path == dst_path:
        logger.info("源路径和目标路径相同，跳过复制: %s", versioned_path)
    elif Path(onnx_path).exists():
        shutil.copy2(onnx_path, str(versioned_path))
        logger.info("ONNX 已复制: %s → %s", onnx_path, versioned_path)
    else:
        logger.warning("ONNX 文件不存在，创建占位文件: %s", versioned_path)
        versioned_path.write_text("")  # Placeholder

    # 2. Copy as campplus.onnx (hot-reload file)
    latest_path = models_dir / "campplus.onnx"
    if versioned_path.exists() and versioned_path.stat().st_size > 0:
        shutil.copy2(str(versioned_path), str(latest_path))
        logger.info("已更新最新模型: %s → %s", versioned_path, latest_path)
    else:
        logger.warning("跳过复制 campplus.onnx: 源文件无效")

    return str(versioned_path)


def register_model_version(
    db_path: str,
    *,
    version: str,
    eval_metric: str,
    eval_value: float,
    prev_eval_value: Optional[float],
    improved: bool,
    train_recording_count: int,
    train_speaker_count: int,
    train_time_sec: float,
    model_path: str,
    notes: Optional[str] = None,
) -> int:
    """
    在 SQLite 中注册模型版本。

    将新版本设为活跃，并将旧版本取消活跃。

    Returns:
        新版本的 ID。
    """
    conn = get_connection(db_path)

    # Deactivate current active model
    active = get_active_model(conn)
    if active:
        deactivate_model(conn, active["version"])

    # Insert new version
    model_md5 = compute_md5(model_path) if Path(model_path).exists() else None
    prev_version = active["version"] if active else None

    version_id = insert_model_version(
        conn,
        version=version,
        eval_metric=eval_metric,
        eval_value=eval_value,
        prev_eval_value=prev_eval_value,
        improved=improved,
        train_recording_count=train_recording_count,
        train_speaker_count=train_speaker_count,
        train_time_sec=train_time_sec,
        model_path=model_path,
        model_md5=model_md5,
        notes=notes,
    )

    conn.close()
    logger.info(
        "模型版本已注册: %s (EER=%.3f%%, improved=%s, path=%s)",
        version, eval_value, improved, model_path,
    )
    return version_id
