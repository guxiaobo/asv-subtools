"""
SQLite 数据库服务（异步版，供 FastAPI 端使用）。

每个请求独立打开/关闭数据库连接，避免共享连接导致的事务交错风险。
SQLite WAL 模式 + busy_timeout 由 init_db 全局设置（schema 创建）。

兼容 train/db.py 表结构。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("asv-api.recording_db")


def _get_default_db_path() -> Path:
    """默认数据库路径：api/services/../../data/training.db"""
    return Path(__file__).resolve().parent.parent.parent / "data" / "training.db"


# ---------------------------------------------------------------------------
# Schema initialization (called once at startup, NOT per request)
# ---------------------------------------------------------------------------

async def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database using unified DDL from data.database (shared with train layer)."""
    import aiosqlite

    path = Path(db_path) if db_path else _get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(path)) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = aiosqlite.Row

        from data.database import init_schema_async
        await init_schema_async(conn)

        # file_hash column fallback (pre-migration DBs)
        try:
            await conn.execute("ALTER TABLE recordings ADD COLUMN file_hash TEXT")
        except Exception:
            pass
        try:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings_file_hash "
                "ON recordings(file_hash) WHERE file_hash IS NOT NULL"
            )
        except Exception:
            pass

        # Legacy model_versions migration
        try:
            rows = await conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='model_versions'"
            )
            if rows:
                pragma = await conn.execute_fetchall("PRAGMA table_info(model_versions)")
                col_names = {r[1] for r in pragma}
                if "version" in col_names and "model_name" not in col_names:
                    await conn.execute("ALTER TABLE model_versions RENAME TO model_versions_legacy")
                    from data.database import DDL_TABLES
                    await conn.execute(DDL_TABLES["model_versions"])
                    logger.info("Schema migration: legacy model_versions renamed to model_versions_legacy")
        except Exception as exc:
            logger.warning("Schema migration check failed: %s", exc)

        logger.info("Database schema initialized: %s", path)
async def _open_conn(db_path: Optional[str] = None) -> "aiosqlite.Connection":
    """打开一个新的独立数据库连接。"""
    import aiosqlite

    path = Path(db_path) if db_path else _get_default_db_path()
    conn = await aiosqlite.connect(str(path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = aiosqlite.Row
    return conn


# ---------------------------------------------------------------------------
# Recording CRUD
# ---------------------------------------------------------------------------


async def get_recording_by_hash(
    file_hash: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """通过文件 MD5 hash 查询已存在的录音记录。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT id, call_id, customer_phone, call_timestamp, local_audio_path, status "
            "FROM recordings WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def insert_recording(
    *,
    biz_system: str,
    call_id: str,
    agent_id: str,
    customer_phone: str,
    call_timestamp: str,
    audio_source_type: str,
    local_audio_path: Optional[str] = None,
    audio_original_url: Optional[str] = None,
    file_hash: Optional[str] = None,
    channel_separated: bool = False,
    duration_sec: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    插入一条录音记录。每个请求独立开/关连接。

    使用 UPSERT 保证 call_id 幂等性：
    - 首次推送 → INSERT
    - 重复推送 → ON CONFLICT DO UPDATE（覆盖 metadata 和文件路径）

    Args:
        db_path: 数据库路径，None 则使用默认路径。
    Returns: recording ID.
    """
    conn = await _open_conn(db_path)
    try:
        from datetime import datetime

        now = datetime.now().isoformat()

        await conn.execute(
            """
            INSERT INTO recordings
                (biz_system, call_id, agent_id, customer_phone, call_timestamp,
                 audio_source_type, local_audio_path, audio_original_url, file_hash,
                 channel_separated, duration_sec, status, pre_status, train_status,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'raw', 'pending', 'pending', ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET
                agent_id = excluded.agent_id,
                customer_phone = excluded.customer_phone,
                call_timestamp = excluded.call_timestamp,
                audio_source_type = excluded.audio_source_type,
                local_audio_path = excluded.local_audio_path,
                audio_original_url = excluded.audio_original_url,
                file_hash = excluded.file_hash,
                channel_separated = excluded.channel_separated,
                duration_sec = excluded.duration_sec,
                updated_at = excluded.updated_at
            """,
            (
                biz_system,
                call_id,
                agent_id,
                customer_phone,
                call_timestamp,
                audio_source_type,
                local_audio_path,
                audio_original_url,
                file_hash,
                1 if channel_separated else 0,
                duration_sec,
                now,
                now,
            ),
        )
        await conn.commit()

        cursor = await conn.execute(
            "SELECT id FROM recordings WHERE call_id = ?", (call_id,)
        )
        row = await cursor.fetchone()
        return row["id"] if row else -1
    finally:
        await conn.close()


async def get_recording(
    call_id: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """通过 call_id 查询录音。每个请求独立连接。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT * FROM recordings WHERE call_id = ?", (call_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def count_pending_preprocess(
    db_path: Optional[str] = None,
) -> int:
    """统计待预处理的录音数。每个请求独立连接。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'pending'"
        )
        row = await cursor.fetchone()
        return row["cnt"]
    finally:
        await conn.close()


async def count_pending_train(db_path: Optional[str] = None) -> int:
    """统计待训练的录音数。每个请求独立连接。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings "
            "WHERE status = 'preprocessed' AND train_status = 'pending'"
        )
        row = await cursor.fetchone()
        return row["cnt"]
    finally:
        await conn.close()


async def list_recordings(
    *,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "id DESC",
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    查询录音列表。支持按坐席 ID、状态筛选。

    Args:
        agent_id: 筛选该坐席的录音，None=所有
        status: 筛选状态，None=所有
        limit: 返回条数
        offset: 偏移
        order_by: 排序字段
        db_path: 数据库路径
    Returns: 录音记录列表
    """
    conn = await _open_conn(db_path)
    try:
        where_clauses = []
        params = []

        if agent_id:
            where_clauses.append("agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("status = ?")
            params.append(status)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cursor = await conn.execute(
            f"SELECT * FROM recordings{where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def update_recording_status(
    call_id: str,
    *,
    status: Optional[str] = None,
    pre_status: Optional[str] = None,
    train_status: Optional[str] = None,
    pre_result: Optional[str] = None,
    pre_error: Optional[str] = None,
    train_result: Optional[str] = None,
    train_error: Optional[str] = None,
    model_version: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """更新录音记录的状态字段。"""
    conn = await _open_conn(db_path)
    try:
        from datetime import datetime

        now = datetime.now().isoformat()
        updates = ["updated_at = ?"]
        params = [now]

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if pre_status is not None:
            updates.append("pre_status = ?")
            params.append(pre_status)
        if train_status is not None:
            updates.append("train_status = ?")
            params.append(train_status)
        if pre_result is not None:
            updates.append("pre_result = ?")
            params.append(pre_result)
        if pre_error is not None:
            updates.append("pre_error = ?")
            params.append(pre_error)
        if train_result is not None:
            updates.append("train_result = ?")
            params.append(train_result)
        if train_error is not None:
            updates.append("train_error = ?")
            params.append(train_error)
        if model_version is not None:
            updates.append("model_version = ?")
            params.append(model_version)
        if status is not None and status in ("preprocessed", "error"):
            updates.append("pre_finished_at = ?")
            params.append(now)
        if train_status is not None and train_status in ("done", "error"):
            updates.append("train_finished_at = ?")
            params.append(now)

        params.append(call_id)
        await conn.execute(
            f"UPDATE recordings SET {', '.join(updates)} WHERE call_id = ?", params
        )
        await conn.commit()
        return True
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


async def get_user_by_username(
    username: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """通过用户名查询用户。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def get_user_by_id(
    user_id: int, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """通过 ID 查询用户。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def list_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出所有用户。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT id, username, role, agent_id, display_name, enabled, "
            "created_at, updated_at FROM users ORDER BY id ASC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def create_user(
    *,
    username: str,
    password_hash: str,
    role: str,
    agent_id: str = "",
    display_name: str = "",
    enabled: bool = True,
    db_path: Optional[str] = None,
) -> int:
    """
    创建用户。

    Returns: 用户 ID.
    Raises: ValueError if username already exists.
    """
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "INSERT INTO users (username, password_hash, role, agent_id, display_name, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, role, agent_id, display_name, 1 if enabled else 0),
        )
        await conn.commit()
        return cursor.lastrowid
    except Exception as e:
        if "UNIQUE" in str(e):
            raise ValueError(f"用户名 '{username}' 已存在")
        raise
    finally:
        await conn.close()


async def update_user(
    user_id: int,
    *,
    password_hash: Optional[str] = None,
    role: Optional[str] = None,
    agent_id: Optional[str] = None,
    display_name: Optional[str] = None,
    enabled: Optional[bool] = None,
    db_path: Optional[str] = None,
) -> bool:
    """更新用户信息。"""
    conn = await _open_conn(db_path)
    try:
        from datetime import datetime

        now = datetime.now().isoformat()
        updates = ["updated_at = ?"]
        params = [now]

        if password_hash is not None:
            updates.append("password_hash = ?")
            params.append(password_hash)
        if role is not None:
            updates.append("role = ?")
            params.append(role)
        if agent_id is not None:
            updates.append("agent_id = ?")
            params.append(agent_id)
        if display_name is not None:
            updates.append("display_name = ?")
            params.append(display_name)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)

        params.append(user_id)
        await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params
        )
        await conn.commit()
        return True
    finally:
        await conn.close()


async def get_users_by_role(
    role: str, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """按角色查询用户列表。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT id, username, agent_id, display_name FROM users "
            "WHERE role = ? AND enabled = 1 ORDER BY id ASC",
            (role,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# Audio Segment CRUD
# ---------------------------------------------------------------------------

async def get_segments_by_recording(
    recording_id: int,
    batch_id: str = "",
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """获取指定录音的所有断句片段。"""
    conn = await _open_conn(db_path)
    try:
        if batch_id:
            cursor = await conn.execute(
                "SELECT * FROM audio_segments "
                "WHERE recording_id = ? AND batch_id = ? ORDER BY segment_index ASC",
                (recording_id, batch_id),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM audio_segments "
                "WHERE recording_id = ? ORDER BY batch_id DESC, segment_index ASC",
                (recording_id,),
            )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()
        await conn.close()


async def get_latest_batch_for_recording(
    recording_id: int,
    db_path: Optional[str] = None,
) -> str:
    """获取录音最新的断句 batch_id。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT batch_id FROM audio_segments "
            "WHERE recording_id = ? ORDER BY id DESC LIMIT 1",
            (recording_id,),
        )
        row = await cursor.fetchone()
        return row["batch_id"] if row else ""
    finally:
        await conn.close()


async def insert_segment(
    *,
    recording_id: int,
    segment_index: int,
    batch_id: str,
    file_path: str,
    start_sec: float = 0,
    end_sec: float = 0,
    duration_sec: float = 0,
    db_path: Optional[str] = None,
) -> int:
    """插入一条断句记录。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "INSERT INTO audio_segments "
            "(recording_id, segment_index, batch_id, file_path, start_sec, end_sec, duration_sec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (recording_id, segment_index, batch_id, file_path, start_sec, end_sec, duration_sec),
        )
        await conn.commit()
        return cursor.lastrowid
    finally:
        await conn.close()


async def update_segment_label(
    seg_id: int,
    *,
    speaker_label: str = "",
    speaker_type: str = "",
    label_source: str = "",
    db_path: Optional[str] = None,
) -> bool:
    """更新片段的说话人标签信息。"""
    conn = await _open_conn(db_path)
    try:
        updates = []
        params = []
        if speaker_label:
            updates.append("speaker_label = ?")
            params.append(speaker_label)
        if speaker_type:
            updates.append("speaker_type = ?")
            params.append(speaker_type)
        if label_source:
            updates.append("label_source = ?")
            params.append(label_source)
        if not updates:
            return False
        params.append(seg_id)
        await conn.execute(
            f"UPDATE audio_segments SET {', '.join(updates)} WHERE id = ?", params
        )
        await conn.commit()
        return True
    finally:
        await conn.close()


async def set_segment_ignored(
    seg_id: int,
    ignored: bool = True,
    db_path: Optional[str] = None,
) -> bool:
    """设置片段是否忽略。"""
    conn = await _open_conn(db_path)
    try:
        await conn.execute(
            "UPDATE audio_segments SET is_ignored = ? WHERE id = ?",
            (1 if ignored else 0, seg_id),
        )
        await conn.commit()
        return True
    finally:
        await conn.close()


async def count_pending_segment_recordings(
    db_path: Optional[str] = None,
) -> int:
    """统计尚未断句的录音数量。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings r "
            "WHERE pre_status = 'done' AND NOT EXISTS ("
            "  SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id"
            ")"
        )
        row = await cursor.fetchone()
        return row["cnt"]
    finally:
        await conn.close()


async def get_batches_for_recording(
    recording_id: int,
    db_path: Optional[str] = None,
) -> List[str]:
    """获取录音的所有断句 batch_id 列表。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT DISTINCT batch_id FROM audio_segments "
            "WHERE recording_id = ? ORDER BY batch_id ASC",
            (recording_id,),
        )
        return [row["batch_id"] for row in await cursor.fetchall()]
    finally:
        await conn.close()


async def list_recordings_with_segments(
    agent_id: str = "",
    customer_phone: str = "",
    date_from: str = "",
    date_to: str = "",
    status_filter: str = "",
    limit: int = 200,
    offset: int = 0,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """查询带有断句信息的录音列表（用于模型管理员页面展示）。"""
    conditions = ["1=1"]
    params: List[Any] = []

    if agent_id:
        conditions.append("r.agent_id = ?")
        params.append(agent_id)
    if customer_phone:
        conditions.append("r.customer_phone = ?")
        params.append(customer_phone)
    if date_from:
        conditions.append("r.call_timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("r.call_timestamp <= ?")
        params.append(date_to + " 23:59:59")
    if status_filter == "pending_segment":
        conditions.append("r.pre_status = 'done'")
        conditions.append(
            "NOT EXISTS (SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id)"
        )
    elif status_filter == "segmented":
        conditions.append(
            "EXISTS (SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id)"
        )
    elif status_filter:
        conditions.append("r.pre_status = ?")
        params.append(status_filter)

    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            f"SELECT r.*, "
            f"(SELECT COUNT(*) FROM audio_segments s WHERE s.recording_id = r.id) AS seg_count, "
            f"(SELECT COUNT(*) FROM audio_segments s WHERE s.recording_id = r.id AND s.is_ignored = 1) AS seg_ignored, "
            f"(SELECT s2.batch_id FROM audio_segments s2 WHERE s2.recording_id = r.id ORDER BY s2.id DESC LIMIT 1) AS latest_batch "
            f"FROM recordings r WHERE {' AND '.join(conditions)} "
            f"ORDER BY r.id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()


async def list_recordings_with_segments_paginated(
    agent_id: str = "",
    customer_phone: str = "",
    date_from: str = "",
    date_to: str = "",
    status_filter: str = "",
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """分页查询录音列表，返回 (recordings, total_count)。"""
    conditions = ["1=1"]
    params: List[Any] = []

    if agent_id:
        conditions.append("r.agent_id = ?")
        params.append(agent_id)
    if customer_phone:
        conditions.append("r.customer_phone = ?")
        params.append(customer_phone)
    if date_from:
        conditions.append("r.call_timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("r.call_timestamp <= ?")
        params.append(date_to + " 23:59:59")
    if status_filter == "pending_segment":
        conditions.append("r.pre_status = 'done'")
        conditions.append(
            "NOT EXISTS (SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id)"
        )
    elif status_filter == "segmented":
        conditions.append(
            "EXISTS (SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id)"
        )
    elif status_filter:
        conditions.append("r.pre_status = ?")
        params.append(status_filter)

    conn = await _open_conn(db_path)
    try:
        # 总计数
        cursor = await conn.execute(
            f"SELECT COUNT(*) FROM recordings r WHERE {' AND '.join(conditions)}",
            params,
        )
        total_count = (await cursor.fetchone())[0]

        # 分页数据
        cursor = await conn.execute(
            f"SELECT r.*, "
            f"(SELECT COUNT(*) FROM audio_segments s WHERE s.recording_id = r.id) AS seg_count, "
            f"(SELECT COUNT(*) FROM audio_segments s WHERE s.recording_id = r.id AND s.is_ignored = 1) AS seg_ignored, "
            f"(SELECT s2.batch_id FROM audio_segments s2 WHERE s2.recording_id = r.id ORDER BY s2.id DESC LIMIT 1) AS latest_batch "
            f"FROM recordings r WHERE {' AND '.join(conditions)} "
            f"ORDER BY r.id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        return rows, total_count
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Model Versions — training version metadata
# --------...[truncated]# but heredocs aren't working...



# ---------------------------------------------------------------------------
# Model Versions — training version metadata
# ---------------------------------------------------------------------------

async def list_model_versions(model_name: str = "", limit: int = 50, offset: int = 0):
    """列出所有模型训练版本。"""
    conn = await _open_conn()
    try:
        params: list = []
        where = ""
        if model_name:
            where = "WHERE model_name = ?"
            params.append(model_name)
        cursor = await conn.execute(
            f"SELECT * FROM model_versions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()


async def create_model_version(model_name: str, version_tag: str, base_model: str,
                               embedding_dim: int, config: str = "{}",
                               metrics: str = "{}", score: float = None,
                               status: str = "training"):
    """创建新的模型版本记录。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            """INSERT INTO model_versions 
               (model_name, version_tag, base_model, embedding_dim, config, 
                metrics, score, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))""",
            (model_name, version_tag, base_model, embedding_dim,
             config, metrics, score, status),
        )
        await conn.commit()
        return cursor.lastrowid
    finally:
        await conn.close()


async def update_model_version(version_id: int, **kwargs):
    """更新模型版本记录的字段。"""
    allowed = {"status", "metrics", "score", "config", "version_tag"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    conn = await _open_conn()
    try:
        set_clauses = []
        params = []
        for k, v in updates.items():
            set_clauses.append(f"{k} = ?")
            params.append(v)
        params.append(version_id)
        await conn.execute(
            f"UPDATE model_versions SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_published_version(model_name: str):
    """获取当前发布的模型版本。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM model_versions WHERE model_name = ? AND status = 'published' ORDER BY id DESC LIMIT 1",
            (model_name,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def get_version_diff(version_a_id: int, version_b_id: int):
    """获取两个版本的对比详情。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            "SELECT * FROM model_versions WHERE id IN (?, ?)", (version_a_id, version_b_id)
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        return rows
    finally:
        await conn.close()


async def count_pending_training():
    """统计待训练的录音数。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'done' AND train_status != 'done'"
        )
        row = await cursor.fetchone()
        return row[0]
    finally:
        await conn.close()


async def list_checkpoints(model_name: str = "", limit: int = 100):
    """列出所有 checkpoints。"""
    conn = await _open_conn()
    try:
        params: list = []
        where = ""
        if model_name:
            where = "WHERE model_name = ?"
            params.append(model_name)
        cursor = await conn.execute(
            f"SELECT * FROM checkpoints {where} ORDER BY id DESC LIMIT ?",
            params + [limit],
        )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()


async def add_checkpoint(model_name: str, version_tag: str, file_path: str,
                         file_size: int = 0, embedding_dim: int = 192,
                         metrics: str = "{}"):
    """添加 checkpoint 记录。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            """INSERT INTO checkpoints 
               (model_name, version_tag, file_path, file_size, embedding_dim,
                metrics, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now','localtime'))""",
            (model_name, version_tag, file_path, file_size, embedding_dim, metrics),
        )
        await conn.commit()
        return cursor.lastrowid
    finally:
        await conn.close()


async def set_published_checkpoint(checkpoint_id: int):
    """发布指定 checkpoint（取消该模型其他已发布检查点）。"""
    conn = await _open_conn()
    try:
        cursor = await conn.execute(
            "SELECT model_name FROM checkpoints WHERE id = ?", (checkpoint_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return False
        model_name = row["model_name"]
        await conn.execute(
            "UPDATE checkpoints SET is_published = 0 WHERE model_name = ?",
            (model_name,),
        )
        await conn.execute(
            "UPDATE checkpoints SET is_published = 1 WHERE id = ?",
            (checkpoint_id,),
        )
        await conn.commit()
        return True
    finally:
        await conn.close()


async def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """获取首页仪表盘统计（单连接批量查询）。"""
    import time as time_module
    import psutil

    conn = await _open_conn(db_path)
    try:
        stats: Dict[str, Any] = {}

        # ── 录音信息 ──
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'pending'"
        )
        stats["pending_vad"] = (await cur.fetchone())["cnt"]

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings "
            "WHERE created_at >= datetime('now', '-7 days')"
        )
        stats["new_7d"] = (await cur.fetchone())["cnt"]

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'done'"
        )
        stats["segmented_recordings"] = (await cur.fetchone())["cnt"]

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE is_ignored IS NULL OR is_ignored = 0"
        )
        stats["total_segments"] = (await cur.fetchone())["cnt"]

        # ── 说话人信息 ──
        cur = await conn.execute(
            "SELECT COUNT(DISTINCT speaker_label) AS cnt FROM audio_segments "
            "WHERE speaker_label IS NOT NULL AND speaker_label != '' "
            "AND (is_ignored IS NULL OR is_ignored = 0)"
        )
        stats["labeled_speakers"] = (await cur.fetchone())["cnt"]

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE speaker_label IS NOT NULL AND speaker_label != '' "
            "AND (is_ignored IS NULL OR is_ignored = 0)"
        )
        stats["labeled_segments"] = (await cur.fetchone())["cnt"]

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE (speaker_label IS NULL OR speaker_label = '') "
            "AND (is_ignored IS NULL OR is_ignored = 0)"
        )
        stats["unlabeled_segments"] = (await cur.fetchone())["cnt"]

        # ── 模型信息 ──
        # 系统只有三个 PyTorch 模型定义（CAM++ / ECAPA / ResNet34）。
        # 每个 model_name 各有一个预训练 checkpoint + 若干增量训练 checkpoint。
        # version_count 直接从 checkpoints 表按 model_name 分组取得。
        CANONICAL_MODELS = ("CAM++", "ECAPA", "ResNet34")

        cur = await conn.execute(
            "SELECT model_name, COUNT(*) AS version_count "
            "FROM checkpoints GROUP BY model_name"
        )
        db_models = {r["model_name"]: r["version_count"] for r in await cur.fetchall()}

        merged = [
            {"model_name": name, "version_count": db_models.get(name, 0)}
            for name in CANONICAL_MODELS
        ]
        stats["models"] = merged
        stats["total_models"] = len(CANONICAL_MODELS)

        # 文件系统 ONNX/PT 文件仅作诊断用，不计入模型定义列表
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        stats["fs_model_files"] = sorted(
            [f for f in (os.listdir(models_dir) if os.path.isdir(models_dir) else [])
             if f.endswith((".onnx", ".pt", ".pth", ".bin", ".gguf"))]
        )

        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments s "
            "JOIN recordings r ON s.recording_id = r.id "
            "WHERE r.pre_status = 'done' "
            "AND (r.train_status IS NULL OR r.train_status NOT IN ('done','training')) "
            "AND (s.is_ignored IS NULL OR s.is_ignored = 0)"
        )
        stats["pending_train_segments"] = (await cur.fetchone())["cnt"]

        # ── 系统运行时 ──
        p = psutil.Process()
        stats["uptime_sec"] = time_module.time() - p.create_time()
        stats["memory_mb"] = round(p.memory_info().rss / 1024 / 1024, 1)
        stats["thread_count"] = p.num_threads()
        stats["cpu_percent"] = p.cpu_percent(interval=0.1)

        return stats
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API Call Logging — real-time call tracking
# ---------------------------------------------------------------------------


async def log_api_call(
    *,
    endpoint: str,
    audio_a_source: str,
    audio_a_value: str = "",
    audio_b_source: str,
    audio_b_value: str = "",
    has_audio_data: bool = False,
    duration_ms: Optional[int] = None,
    score: Optional[float] = None,
    decision: Optional[str] = None,
    threshold: Optional[float] = None,
    scenario: Optional[str] = None,
    caller_ip: Optional[str] = None,
    error_detail: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """记录一次 API 调用日志。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            """INSERT INTO api_call_logs
               (endpoint, audio_a_source, audio_a_value, audio_b_source,
                audio_b_value, has_audio_data, duration_ms, score,
                decision, threshold, scenario, caller_ip, error_detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                endpoint,
                audio_a_source,
                audio_a_value,
                audio_b_source,
                audio_b_value,
                1 if has_audio_data else 0,
                duration_ms,
                score,
                decision,
                threshold,
                scenario,
                caller_ip,
                error_detail,
            ),
        )
        await conn.commit()
        return cursor.lastrowid
    finally:
        await conn.close()


async def get_api_call_stats(days: int = 1, db_path: Optional[str] = None) -> dict:
    """获取最近 N 天的 API 调用统计。

    Returns:
        dict with keys: total_calls, avg_duration_ms, max_duration_ms,
        p50_ms, p95_ms, success_count, fail_count
    """
    conn = await _open_conn(db_path)
    try:
        import sqlite3

        result: dict = {
            "total_calls": 0,
            "avg_duration_ms": 0,
            "max_duration_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "success_count": 0,
            "fail_count": 0,
        }

        cur = await conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(duration_ms),0), "
            "COALESCE(MAX(duration_ms),0) "
            "FROM api_call_logs "
            "WHERE created_at >= datetime('now', ? || ' days', 'localtime')",
            (f"-{days}",),
        )
        row = await cur.fetchone()
        result["total_calls"] = row[0]
        result["avg_duration_ms"] = round(row[1], 1) if row[1] else 0
        result["max_duration_ms"] = row[2] or 0

        # success/fail (where score IS NOT NULL = success)
        cur = await conn.execute(
            "SELECT COUNT(*) FROM api_call_logs "
            "WHERE created_at >= datetime('now', ? || ' days', 'localtime') "
            "AND score IS NOT NULL",
            (f"-{days}",),
        )
        result["success_count"] = (await cur.fetchone())[0]

        cur = await conn.execute(
            "SELECT COUNT(*) FROM api_call_logs "
            "WHERE created_at >= datetime('now', ? || ' days', 'localtime') "
            "AND error_detail IS NOT NULL",
            (f"-{days}",),
        )
        result["fail_count"] = (await cur.fetchone())[0]

        # Percentiles via subquery with sorting
        cur = await conn.execute(
            "SELECT duration_ms FROM api_call_logs "
            "WHERE created_at >= datetime('now', ? || ' days', 'localtime') "
            "AND duration_ms IS NOT NULL ORDER BY duration_ms",
            (f"-{days}",),
        )
        values = [r[0] for r in await cur.fetchall()]
        if values:
            n = len(values)
            idx50 = max(0, int(n * 0.5) - 1)
            idx95 = max(0, int(n * 0.95) - 1)
            result["p50_ms"] = values[idx50]
            result["p95_ms"] = values[idx95]

        return result
    finally:
        await conn.close()


async def get_multi_period_call_stats(db_path: Optional[str] = None) -> dict:
    """获取多维 API 调用统计（当天 / 7天 / 30天）。"""
    return {
        "today": await get_api_call_stats(1, db_path),
        "last_7d": await get_api_call_stats(7, db_path),
        "last_30d": await get_api_call_stats(30, db_path),
    }


# ---------------------------------------------------------------------------
# Segment labeling — history tracking & stats
# ---------------------------------------------------------------------------


async def log_segment_label_change(
    segment_id: int,
    *,
    old_label: str = "",
    new_label: str = "",
    old_type: str = "",
    new_type: str = "",
    old_ignored: int = 0,
    new_ignored: int = 0,
    operated_by: str = "",
    auto_reason: str = "",
    db_path: Optional[str] = None,
) -> int:
    """记录一次片段标签变更历史。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            """INSERT INTO segment_label_history
               (segment_id, old_label, new_label, old_type, new_type,
                old_ignored, new_ignored, operated_by, auto_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (segment_id, old_label, new_label, old_type, new_type,
             old_ignored, new_ignored, operated_by, auto_reason),
        )
        await conn.commit()
        return cursor.lastrowid
    finally:
        await conn.close()


async def get_segment_by_id(
    segment_id: int,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """获取单个片段详情。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT * FROM audio_segments WHERE id = ?", (segment_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def get_segment_label_history(
    segment_id: int,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """获取指定片段的标签变更历史。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT * FROM segment_label_history "
            "WHERE segment_id = ? ORDER BY id DESC LIMIT ?",
            (segment_id, limit),
        )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()


async def get_segment_stats(db_path: Optional[str] = None) -> Dict[str, int]:
    """获取录音断句/打标汇总统计（用于 segments 页面顶部指标栏）。

    Returns:
        dict with keys:
          total_recordings   — 总录音数（已预处理完成的）
          segmented          — 已断句录音数
          labeled            — 已打标片段数
          unsegmented        — 未断句录音数（已预处理但无片段）
          unlabeled          — 未打标片段数
    """
    conn = await _open_conn(db_path)
    try:
        stats: Dict[str, int] = {}

        # 总录音数（已预处理完成）
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'done'"
        )
        stats["total_recordings"] = (await cur.fetchone())["cnt"]

        # 已断句（有片段数据的录音数）
        cur = await conn.execute(
            "SELECT COUNT(DISTINCT recording_id) AS cnt FROM audio_segments"
        )
        stats["segmented"] = (await cur.fetchone())["cnt"]

        # 未断句（已预处理但无片段）
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM recordings r "
            "WHERE r.pre_status = 'done' AND NOT EXISTS ("
            "  SELECT 1 FROM audio_segments s WHERE s.recording_id = r.id"
            ")"
        )
        stats["unsegmented"] = (await cur.fetchone())["cnt"]

        # 已打标片段
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE speaker_label IS NOT NULL AND speaker_label != '' "
            "AND (is_ignored IS NULL OR is_ignored = 0)"
        )
        stats["labeled"] = (await cur.fetchone())["cnt"]

        # 未打标片段
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE (speaker_label IS NULL OR speaker_label = '') "
            "AND (is_ignored IS NULL OR is_ignored = 0)"
        )
        stats["unlabeled"] = (await cur.fetchone())["cnt"]

        return stats
    finally:
        await conn.close()


async def update_segment_trained_status(
    segment_ids: List[int],
    status: str = "trained",
    db_path: Optional[str] = None,
) -> int:
    """批量更新片段的训练状态。

    Args:
        segment_ids: 片段 ID 列表
        status: 'untrained' | 'training' | 'trained'
    Returns:
        更新的行数
    """
    if not segment_ids:
        return 0
    conn = await _open_conn(db_path)
    try:
        placeholders = ",".join("?" for _ in segment_ids)
        cursor = await conn.execute(
            f"UPDATE audio_segments SET trained_status = ? "
            f"WHERE id IN ({placeholders})",
            [status] + segment_ids,
        )
        await conn.commit()
        return cursor.rowcount
    finally:
        await conn.close()


async def list_speakers_from_segments(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从已打标片段中提取说话人列表（去重）。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT speaker_label AS speaker_id, COUNT(*) AS seg_count, "
            "speaker_type, MAX(created_at) AS last_labeled "
            "FROM audio_segments "
            "WHERE speaker_label IS NOT NULL AND speaker_label != '' "
            "AND (is_ignored IS NULL OR is_ignored = 0) "
            "GROUP BY speaker_label ORDER BY seg_count DESC"
        )
        return [dict(r) for r in await cursor.fetchall()]
    finally:
        await conn.close()


async def count_labeled_segments_for_speakers(
    db_path: Optional[str] = None,
) -> int:
    """统计涉及已打标说话人的片段数（跟 labeled_speakers 配合展示）。"""
    conn = await _open_conn(db_path)
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM audio_segments "
            "WHERE speaker_label IS NOT NULL AND speaker_label != '' "
            "AND (is_ignored IS NULL OR is_ignored = 0) "
            "AND trained_status = 'untrained'"
        )
        return (await cursor.fetchone())["cnt"]
    finally:
        await conn.close()
