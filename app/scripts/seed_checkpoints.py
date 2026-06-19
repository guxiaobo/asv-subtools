"""
种子脚本：扫描磁盘上所有现有的模型 checkpoint，注册到 model_versions 和 checkpoints 表。

创建系统所需的模型目录结构并记录演化历史。

运行: python scripts/seed_checkpoints.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_checkpoints")

# 项目路径（所有元数据路径从此基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # asv-subtools/
APP_DIR = PROJECT_ROOT / "app"  # app/
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "training.db"


def to_rel(path: Path) -> str:
    """将绝对路径转换为相对 PROJECT_ROOT 的路径。"""
    return str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path)


# 旧目录（向后兼容保留）
old_weights_dir = APP_DIR / "pytorch_weights"
old_fine_tuned_dir = old_weights_dir / "fine_tuned"

# ===================================================================
# 定义公开预训练 checkpoint（作为演化根节点）
# ===================================================================
PRETRAINED = {
    "CAM++": {
        "file": old_weights_dir / "campplus_cn_common.pt",
        "version": "v0_pretrained",
        "dim": 192,
        "desc": "CAM++ 原始公开预训练 checkpoint (CN-Celeb)",
        "base": "",
    },
    "ECAPA": {
        "file": old_weights_dir / "avg_model.pt",
        "version": "v0_pretrained",
        "dim": 192,
        "desc": "ECAPA-TDNN 原始公开预训练 checkpoint (VoxCeleb1+2)",
        "base": "",
    },
    "ResNet34": {
        # avg_model/ 是一个目录，内含 avg_model.pt
        "file": old_weights_dir / "avg_model",  # single Zip file, no extension
        "version": "v0_pretrained",
        "dim": 256,
        "desc": "ResNet34 原始公开预训练 checkpoint (VoxCeleb1+2)",
        "base": "",
    },
}

# 增量训练按模型分组
FINE_TUNED = {
    "CAM++": [
        ("v1", old_fine_tuned_dir / "campplus_backbone.pt"),
    ],
    "ECAPA": [
        ("v1", old_fine_tuned_dir / "ecapa_backbone.pt"),
    ],
    "ResNet34": [
        ("v1", old_fine_tuned_dir / "resnet_backbone.pt"),
    ],
}

# ===================================================================
# 规范化模型目录结构
# ===================================================================
# app/model_data/
#   checkpoints/
#     CAM++/
#       v0_pretrained/
#         model.pt       (copy/campplus_cn_common.pt)
#         manifest.json
#       v1/
#         backbone.pt
#         manifest.json
#     ECAPA/
#       v0_pretrained/
#         model.pt
#         manifest.json
#       v1/
#         backbone.pt
#         manifest.json
#     ResNet34/
#       v0_pretrained/
#         model.pt
#         manifest.json
#       v1/
#         backbone.pt
#         manifest.json
#   deployed/
#     CAM++/
#       v0.onnx  (copy from api/models/campplus.onnx or link)
#     ECAPA/
#       v0.onnx
#     ResNet34/
#       v0.onnx

MODEL_DATA = APP_DIR / "model_data"
CHECKPOINTS_ROOT = MODEL_DATA / "checkpoints"
DEPLOYED_ROOT = MODEL_DATA / "deployed"


def ensure_dirs():
    """创建规范化目录结构。"""
    for mname in ("CAM++", "ECAPA", "ResNet34"):
        (CHECKPOINTS_ROOT / mname).mkdir(parents=True, exist_ok=True)
        (DEPLOYED_ROOT / mname).mkdir(parents=True, exist_ok=True)


def copy_checkpoint(src: Path, dst_dir: Path, name: str = "model.pt") -> Path:
    """复制 checkpoint 到目标目录（如已存在跳过）。"""
    dst = dst_dir / name
    if dst.exists():
        logger.info("  已存在: %s", dst)
        return dst
    if not src.exists():
        logger.warning("  ⚠  源文件不存在: %s", src)
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("  复制: %s → %s", src, dst)
    return dst


def write_manifest(dst_dir: Path, data: dict):
    """写入 manifest.json。"""
    manifest_path = dst_dir / "manifest.json"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("  manifest: %s", manifest_path)


# ===================================================================
# DB 操作
# ===================================================================


def get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接（创建表如果不存在）。"""
    from data.database import init_schema, get_connection  # noqa: E402

    conn = get_connection(str(DB_PATH))
    init_schema(conn)
    return conn


def existing_version(conn: sqlite3.Connection, version: str) -> bool:
    """检查 version 是否已存在。"""
    c = conn.execute(
        "SELECT COUNT(*) FROM model_versions WHERE version_tag = ?", (version,)
    )
    return c.fetchone()[0] > 0


def insert_into_both_tables(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    version_tag: str,
    file_path: str,
    embedding_dim: int,
    base_model: str,
    score: float = 0.0,
    status: str = "published",
    description: str = "",
):
    """
    同时写入 model_versions 和 checkpoints 表。
    """
    now = datetime.now().isoformat()

    # 1) checkpoints 表
    full_path = PROJECT_ROOT / file_path
    file_size = full_path.stat().st_size if full_path.exists() else 0
    metrics = json.dumps({
        "dim": embedding_dim,
        "base_model": base_model,
        "description": description,
        "files": sorted([str(p) for p in full_path.parent.rglob("*") if p.is_file() and p.suffix not in (".json",)]),
    }, ensure_ascii=False)

    # checkpoints 表（用于断句/训练管理）
    conn.execute(
        """INSERT OR IGNORE INTO checkpoints
           (model_name, version_tag, file_path, file_size, embedding_dim, metrics, is_published, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
        (model_name, version_tag, file_path, file_size, embedding_dim, metrics, now),
    )

    # 2) model_versions 表（用于版本发布/部署管理）
    conn.execute(
        """INSERT OR IGNORE INTO model_versions
           (model_name, version_tag, base_model, embedding_dim, config, metrics, score, status, created_at)
           VALUES (?, ?, ?, ?, '{}', ?, ?, ?, ?)""",
        (model_name, version_tag, base_model, embedding_dim, metrics, score, status, now),
    )
    conn.commit()


# ===================================================================
# 种子流程
# ===================================================================


def seed():
    """主种子流程。"""
    logger.info("=" * 60)
    logger.info("模型 checkpoint 种子脚本")
    logger.info("DB: %s", DB_PATH)
    logger.info("旧权重目录: %s", old_weights_dir)
    logger.info("新规范化目录: %s", MODEL_DATA)
    logger.info("=" * 60)

    conn = get_conn()
    ensure_dirs()
    total = 0

    # ── Phase 1: 公开预训练 checkpoint（演化根节点）──
    logger.info("\n▸ Phase 1: 公开预训练 checkpoint")
    for mname, info in PRETRAINED.items():
        src = info["file"]
        vtag = info["version"]
        dst_dir = CHECKPOINTS_ROOT / mname / vtag

        if existing_version(conn, f"{mname}@{vtag}"):
            logger.info("  已注册: %s@%s", mname, vtag)
            continue

        copied = copy_checkpoint(src, dst_dir)
        if copied is None:
            continue

        write_manifest(dst_dir, {
            "model": mname,
            "version": vtag,
            "description": info["desc"],
            "embedding_dim": info["dim"],
            "source": to_rel(src),
            "created_at": datetime.now().isoformat(),
            "base_model": "",
            "files": [
                {"name": "model.pt", "path": to_rel(copied), "size": copied.stat().st_size},
            ],
        })

        insert_into_both_tables(
            conn,
            model_name=mname,
            version_tag=f"{mname}@{vtag}",
            file_path=to_rel(copied),
            embedding_dim=info["dim"],
            base_model=info["base"],
            score=0.0,
            status="published",
            description=info["desc"],
        )
        total += 1
        logger.info("  ✅ %s %s 注册完成", mname, vtag)

    # ── Phase 2: 增量训练 checkpoint（演化子节点）──
    logger.info("\n▸ Phase 2: 增量训练 checkpoint")
    for mname, versions in FINE_TUNED.items():
        for vtag, src in versions:
            vkey = f"{mname}@{vtag}"
            dst_dir = CHECKPOINTS_ROOT / mname / vtag

            if existing_version(conn, vkey):
                logger.info("  已注册: %s", vkey)
                continue

            copied = copy_checkpoint(src, dst_dir)
            if copied is None:
                continue

            write_manifest(dst_dir, {
                "model": mname,
                "version": vtag,
                "description": f"{mname} 第 {vtag} 轮增量训练 (backbone)",
                "embedding_dim": PRETRAINED[mname]["dim"],
                "source": to_rel(src),
                "created_at": datetime.now().isoformat(),
                "base_model": f"{mname}@v0_pretrained",
                "files": [
                    {"name": "backbone.pt", "path": to_rel(copied), "size": copied.stat().st_size},
                ],
            })

            # score 从 DB 中已有评估数据取（如果有的话）
            insert_into_both_tables(
                conn,
                model_name=mname,
                version_tag=vkey,
                file_path=to_rel(copied),
                embedding_dim=PRETRAINED[mname]["dim"],
                base_model=f"{mname}@v0_pretrained",
                score=0.0,
                status="published",
                description=f"{mname} 第 {vtag} 轮增量训练",
            )
            total += 1
            logger.info("  ✅ %s %s 注册完成", mname, vtag)

    # ── Phase 3: 部署的 ONNX 文件 ──
    logger.info("\n▸ Phase 3: ONNX 部署文件快照")
    onnx_dir = APP_DIR / "api" / "models"
    onnx_map = {
        "CAM++": [
            ("v0", onnx_dir / "campplus.onnx"),
            ("v0_cnceleb", onnx_dir / "voxceleb_CAM++.onnx"),
        ],
        "ECAPA": [
            ("v0", onnx_dir / "ecapa-speaker-v1.onnx"),
        ],
        "ResNet34": [
            ("v0", onnx_dir / "voxceleb_resnet34_LM.onnx"),
        ],
    }
    for mname, onnx_list in onnx_map.items():
        for vtag, src in onnx_list:
            if not src.exists():
                logger.info("  ⚠  ONNX 不存在: %s", src)
                continue
            dst_dir = DEPLOYED_ROOT / mname
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{vtag}.onnx"
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info("  ONNX 复制: %s → %s", src, dst)
            else:
                logger.info("  ONNX 已存在: %s", dst)

    conn.close()
    logger.info("\n✓ 种子脚本完成，共注册 %d 条记录", total)


if __name__ == "__main__":
    seed()
