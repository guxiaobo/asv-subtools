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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("asv-api.recording_db")


def _get_default_db_path() -> Path:
    """默认数据库路径：api/services/../../data/training.db"""
    return Path(__file__).resolve().parent.parent.parent / "data" / "training.db"


# ---------------------------------------------------------------------------
# Schema initialization (called once at startup, NOT per request)
# ---------------------------------------------------------------------------

async def init_db(db_path: Optional[str] = None) -> None:
    """
    初始化 SQLite 数据库（建表 + 优化 pragma）。
    应用启动时调用一次，完成后关闭连接。

    Args:
        db_path: 数据库文件路径，None 则使用默认路径。
    """
    import aiosqlite

    path = Path(db_path) if db_path else _get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(path)) as conn:
        # Pragma 只在此初始化，后续每个连接会单独设置
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = aiosqlite.Row

        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                biz_system      TEXT    NOT NULL,
                call_id         TEXT    NOT NULL UNIQUE,
                agent_id        TEXT    NOT NULL,
                customer_phone  TEXT    NOT NULL,
                call_timestamp  TEXT    NOT NULL,
                channel_separated INTEGER DEFAULT 0,
                duration_sec    REAL,
                audio_source_type TEXT  NOT NULL,
                audio_original_url TEXT,
                local_audio_path    TEXT,
                file_hash       TEXT,
                status          TEXT    NOT NULL DEFAULT 'raw',
                pre_status      TEXT    DEFAULT 'pending',
                pre_result      TEXT,
                pre_error       TEXT,
                pre_finished_at TEXT,
                train_status    TEXT    DEFAULT 'pending',
                train_result    TEXT,
                train_error     TEXT,
                train_finished_at TEXT,
                model_version   TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Add file_hash column for existing databases (safe IF NOT EXISTS)
        for alter_stmt in [
            "ALTER TABLE recordings ADD COLUMN file_hash TEXT",
        ]:
            try:
                await conn.execute(alter_stmt)
            except Exception:
                pass  # Column already exists

        # Unique index for fast duplicate detection by hash
        try:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings_file_hash "
                "ON recordings(file_hash) WHERE file_hash IS NOT NULL"
            )
        except Exception:
            pass
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version         TEXT    NOT NULL UNIQUE,
                eval_metric     TEXT    NOT NULL,
                eval_value      REAL    NOT NULL,
                prev_eval_value REAL,
                improved        INTEGER DEFAULT 0,
                train_recording_count INTEGER NOT NULL,
                train_speaker_count   INTEGER NOT NULL,
                train_time_sec   REAL,
                previous_version TEXT,
                model_path      TEXT    NOT NULL,
                model_md5       TEXT,
                is_active       INTEGER DEFAULT 0,
                notes           TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS speaker_voiceprints (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name      TEXT    NOT NULL,
                speaker_type    TEXT    NOT NULL,
                speaker_id      TEXT    NOT NULL,
                embedding       BLOB    NOT NULL,
                segment_count   INTEGER DEFAULT 1,
                source_call_ids TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        for idx_stmt in [
            "CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_pre_status ON recordings(pre_status)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_train_status ON recordings(train_status)",
            "CREATE INDEX IF NOT EXISTS idx_recordings_biz_agent ON recordings(biz_system, agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active)",
        ]:
            try:
                await conn.execute(idx_stmt)
            except Exception:
                pass

        # ── Users table (auth) ────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE,
                password_hash   TEXT    NOT NULL,
                role            TEXT    NOT NULL DEFAULT 'agent'
                                        CHECK (role IN ('admin','model_manager','agent')),
                agent_id        TEXT    NOT NULL DEFAULT '',
                display_name    TEXT    NOT NULL DEFAULT '',
                enabled         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await conn.commit()

    logger.info("Recording DB schema initialized: %s", path)


# ---------------------------------------------------------------------------
# Connection helpers (per-request)
# ---------------------------------------------------------------------------

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
