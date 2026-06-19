"""
迁移脚本：建立模型架构定义表 model_definitions，修复 checkpoints 重复数据，
回填 checkpoints.model_def_id / base_checkpoint_id / status，并建立
checkpoint_training_segments 关联。

幂等：可重复运行，已存在的记录会被跳过。

运行: python scripts/migrate_model_defs.py
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_model_defs")

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "training.db"
PYTORCH_MODELS_DIR = APP_DIR / "pytorch_models"

# ── 三个 PyTorch 模型定义的元信息 ──
# code_path 相对于 app/ 目录
MODEL_DEFS = [
    {
        "name": "CAM++",
        "arch_version": "v1",
        "code_path": "pytorch_models/campp_model.py",
        "class_name": "CAMPlus",
        "embedding_dim": 192,
        "description": "CAM++ (Concat-Aggregated MFCC Plus Plus)，电话场景分离最佳",
    },
    {
        "name": "ECAPA",
        "arch_version": "v1",
        "code_path": "pytorch_models/ecapa_model.py",
        "class_name": "ECAPA_TDNN",
        "embedding_dim": 192,
        "description": "ECAPA-TDNN，时延神经网络 + 通道注意力",
    },
    {
        "name": "ResNet34",
        "arch_version": "v1",
        "code_path": "pytorch_models/resnet34_model.py",
        "class_name": "ResNet34",
        "embedding_dim": 256,
        "description": "ResNet34 残差网络，256 维 embedding，通用性强",
    },
]


def code_hash(rel_path: str) -> str:
    """计算 PyTorch 模型代码文件的 MD5 hash。"""
    full = APP_DIR / rel_path
    if not full.exists():
        return ""
    return hashlib.md5(full.read_bytes()).hexdigest()[:16]


def get_conn() -> sqlite3.Connection:
    sys.path.insert(0, str(APP_DIR))
    from data.database import init_schema, get_connection

    conn = get_connection(str(DB_PATH))
    init_schema(conn)
    return conn


def dedup_checkpoints(conn: sqlite3.Connection) -> int:
    """去除 checkpoints 表中 (model_name, version_tag, file_path) 完全相同的重复行。"""
    cur = conn.execute(
        "SELECT model_name, version_tag, file_path, MIN(id) AS keep_id "
        "FROM checkpoints GROUP BY model_name, version_tag, file_path "
        "HAVING COUNT(*) > 1"
    )
    dup_groups = cur.fetchall()
    if not dup_groups:
        logger.info("checkpoints 表无重复数据")
        return 0
    total_removed = 0
    for row in dup_groups:
        keep_id = row["keep_id"]
        c = conn.execute(
            "DELETE FROM checkpoints WHERE model_name=? AND version_tag=? AND file_path=? AND id!=?",
            (row["model_name"], row["version_tag"], row["file_path"], keep_id),
        )
        total_removed += c.rowcount
    conn.commit()
    logger.info("去重：删除 %d 行重复 checkpoint 记录", total_removed)
    return total_removed


def seed_model_definitions(conn: sqlite3.Connection) -> dict:
    """填充 model_definitions 表，返回 {name: def_id} 映射。"""
    name_to_id = {}
    for info in MODEL_DEFS:
        ch = code_hash(info["code_path"])
        existing = conn.execute(
            "SELECT id FROM model_definitions WHERE name=? AND arch_version=?",
            (info["name"], info["arch_version"]),
        ).fetchone()
        if existing:
            # 更新 code_hash（代码可能已变化）
            conn.execute(
                "UPDATE model_definitions SET code_hash=?, code_path=?, class_name=? "
                "WHERE id=?",
                (ch, info["code_path"], info["class_name"], existing["id"]),
            )
            name_to_id[info["name"]] = existing["id"]
            logger.info("  更新 model_definitions: %s@%s (id=%d, hash=%s)",
                        info["name"], info["arch_version"], existing["id"], ch)
        else:
            cur = conn.execute(
                "INSERT INTO model_definitions "
                "(name, arch_version, code_path, code_hash, class_name, embedding_dim, description, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (info["name"], info["arch_version"], info["code_path"], ch,
                 info["class_name"], info["embedding_dim"], info["description"], "system"),
            )
            name_to_id[info["name"]] = cur.lastrowid
            logger.info("  新建 model_definitions: %s@%s (id=%d, hash=%s)",
                        info["name"], info["arch_version"], cur.lastrowid, ch)
    conn.commit()
    return name_to_id


def backfill_checkpoints(conn: sqlite3.Connection, name_to_id: dict) -> int:
    """回填 checkpoints.model_def_id / status / base_checkpoint_id。"""
    updated = 0
    # 按 (model_name, version_tag) 分组，建立 tag→id 映射，用于计算 base
    rows = conn.execute(
        "SELECT id, model_name, version_tag, model_def_id, status "
        "FROM checkpoints ORDER BY model_name, version_tag"
    ).fetchall()
    tag_to_id = {(r["model_name"], r["version_tag"]): r["id"] for r in rows}

    for r in rows:
        updates = {}
        if r["model_def_id"] is None:
            def_id = name_to_id.get(r["model_name"])
            if def_id:
                updates["model_def_id"] = def_id
        if not r["status"] or r["status"] == "incremental":
            # version_tag 含 v0_pretrained → pretrained；其余 → incremental
            vtag = r["version_tag"]
            if "v0_pretrained" in vtag:
                updates["status"] = "pretrained"
            else:
                updates["status"] = "incremental"
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            params = list(updates.values()) + [r["id"]]
            conn.execute(f"UPDATE checkpoints SET {set_clause} WHERE id=?", params)
            updated += 1

    # 回填 base_checkpoint_id：增量 checkpoint 的 base 是同模型的 v0_pretrained
    pretrained_map = {}  # model_name → checkpoint_id
    for r in conn.execute(
        "SELECT id, model_name FROM checkpoints WHERE status='pretrained'"
    ).fetchall():
        pretrained_map[r["model_name"]] = r["id"]
    for r in conn.execute(
        "SELECT id, model_name, base_checkpoint_id FROM checkpoints WHERE status='incremental'"
    ).fetchall():
        if r["base_checkpoint_id"] is None:
            base_id = pretrained_map.get(r["model_name"])
            if base_id:
                conn.execute(
                    "UPDATE checkpoints SET base_checkpoint_id=? WHERE id=?",
                    (base_id, r["id"]),
                )

    conn.commit()
    logger.info("回填 checkpoints：%d 行更新", updated)
    return updated


def verify(conn: sqlite3.Connection):
    logger.info("\n═══ 验证 ═══")
    logger.info("model_definitions:")
    for r in conn.execute(
        "SELECT id, name, arch_version, code_path, code_hash, class_name, embedding_dim "
        "FROM model_definitions ORDER BY name"
    ):
        logger.info("  [%d] %s@%s  dim=%d  %s::%s  hash=%s",
                    r["id"], r["name"], r["arch_version"], r["embedding_dim"],
                    r["code_path"], r["class_name"], r["code_hash"])
    logger.info("\ncheckpoints:")
    for r in conn.execute(
        "SELECT id, model_name, version_tag, model_def_id, base_checkpoint_id, status "
        "FROM checkpoints ORDER BY model_name, version_tag"
    ):
        base = f"→#{r['base_checkpoint_id']}" if r["base_checkpoint_id"] else "—"
        logger.info("  [%d] %-8s %-22s def=%s base=%s status=%s",
                    r["id"], r["model_name"], r["version_tag"],
                    r["model_def_id"], base, r["status"])
    logger.info("\ncheckpoint_training_segments: %d 条",
                conn.execute("SELECT COUNT(*) FROM checkpoint_training_segments").fetchone()[0])


def main():
    logger.info("=" * 60)
    logger.info("模型架构定义迁移脚本")
    logger.info("DB: %s", DB_PATH)
    logger.info("=" * 60)

    conn = get_conn()
    removed = dedup_checkpoints(conn)
    name_to_id = seed_model_definitions(conn)
    backfilled = backfill_checkpoints(conn, name_to_id)
    verify(conn)
    conn.close()
    logger.info("\n✓ 迁移完成：去重 %d 行，回填 %d 行", removed, backfilled)


if __name__ == "__main__":
    main()
