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
                 audio_source_type, local_audio_path, audio_original_url,
                 channel_separated, duration_sec, status, pre_status, train_status,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'raw', 'pending', 'pending', ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET
                agent_id = excluded.agent_id,
                customer_phone = excluded.customer_phone,
                call_timestamp = excluded.call_timestamp,
                audio_source_type = excluded.audio_source_type,
                local_audio_path = excluded.local_audio_path,
                audio_original_url = excluded.audio_original_url,
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
