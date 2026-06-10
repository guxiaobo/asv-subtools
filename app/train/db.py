"""
SQLite 数据库操作（同步版，供 train/ CLI 模块使用）。

提供原子性任务领取、状态更新、模型版本管理等核心操作。
API 服务使用异步版 (api/services/recording_db.py)。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from train.schemas import (
    PreprocessResult,
    PreprocessStatus,
    RecordingRow,
    RecordingStatus,
    TrainStatus,
)

logger = logging.getLogger("train.db")

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

# Default database path (relative to project root)
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "training.db"


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """获取 SQLite 连接，已配置 WAL 模式和超时。"""
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

CREATE_RECORDINGS_TABLE = """
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
"""

CREATE_MODEL_VERSIONS_TABLE = """
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
"""

CREATE_SPEAKER_VOICEPRINTS_TABLE = """
CREATE TABLE IF NOT EXISTS speaker_voiceprints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    speaker_type    TEXT    NOT NULL CHECK(speaker_type IN ('agent', 'customer')),
    speaker_id      TEXT    NOT NULL,
    embedding       BLOB   NOT NULL,
    segment_count   INTEGER DEFAULT 1,
    source_call_ids TEXT,   -- JSON array of call_ids used
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status)",
    "CREATE INDEX IF NOT EXISTS idx_recordings_pre_status ON recordings(pre_status)",
    "CREATE INDEX IF NOT EXISTS idx_recordings_train_status ON recordings(train_status)",
    "CREATE INDEX IF NOT EXISTS idx_recordings_biz_agent ON recordings(biz_system, agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active)",
    "CREATE INDEX IF NOT EXISTS idx_speaker_vp_model ON speaker_voiceprints(model_name)",
    "CREATE INDEX IF NOT EXISTS idx_speaker_vp_type ON speaker_voiceprints(speaker_type, speaker_id)",
]


def init_db(conn: sqlite3.Connection) -> None:
    """初始化数据库表结构和索引。"""
    conn.execute(CREATE_RECORDINGS_TABLE)
    conn.execute(CREATE_MODEL_VERSIONS_TABLE)
    conn.execute(CREATE_SPEAKER_VOICEPRINTS_TABLE)
    for idx in CREATE_INDEXES:
        try:
            conn.execute(idx)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    logger.info("Database schema initialized: %s", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Recording operations
# ---------------------------------------------------------------------------

def insert_recording(
    conn: sqlite3.Connection,
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
) -> int:
    """插入一条录音记录。如果 call_id 已存在则更新。"""
    now = __import__("datetime").datetime.now().isoformat()
    conn.execute(
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
            biz_system, call_id, agent_id, customer_phone, call_timestamp,
            audio_source_type, local_audio_path, audio_original_url,
            1 if channel_separated else 0, duration_sec,
            now, now,
        ),
    )
    conn.commit()
    cursor = conn.execute("SELECT id FROM recordings WHERE call_id = ?", (call_id,))
    row = cursor.fetchone()
    return row["id"] if row else -1


def get_recording_by_call_id(conn: sqlite3.Connection, call_id: str) -> Optional[Dict[str, Any]]:
    """通过 call_id 查询录音记录。"""
    cursor = conn.execute("SELECT * FROM recordings WHERE call_id = ?", (call_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_recording_by_id(conn: sqlite3.Connection, rec_id: int) -> Optional[Dict[str, Any]]:
    """通过 id 查询录音记录。"""
    cursor = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Atomic task claiming — prevents duplicate processing
# ---------------------------------------------------------------------------

def claim_pending_recordings(
    conn: sqlite3.Connection,
    limit: int = 100,
    status_field: str = "pre_status",
) -> List[Dict[str, Any]]:
    """
    原子性领取 N 条待处理录音。

    使用 UPDATE ... RETURNING 确保两个进程不会抢到同一条录音。
    status_field: 'pre_status'（预处理）或 'train_status'（训练）。

    Returns:
        成功标记为 processing 的录音记录列表。
    """
    rows = conn.execute(
        f"""
        UPDATE recordings
        SET {status_field} = 'processing',
            updated_at = datetime('now')
        WHERE rowid IN (
            SELECT rowid FROM recordings
            WHERE {status_field} = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
        )
        RETURNING *
        """,
        (limit,),
    ).fetchall()
    conn.commit()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Status updates (preprocessing)
# ---------------------------------------------------------------------------

def update_pre_status(
    conn: sqlite3.Connection,
    rec_id: int,
    pre_status: str,
    pre_result: Optional[Dict] = None,
    pre_error: Optional[str] = None,
) -> None:
    """
    更新录音的预处理状态。

    pre_status='done' 时同时将录音 status 更新为 'preprocessed'。
    """
    now = __import__("datetime").datetime.now().isoformat()
    result_json = json.dumps(pre_result, ensure_ascii=False) if pre_result else None

    conn.execute(
        """
        UPDATE recordings
        SET pre_status = ?,
            pre_result = ?,
            pre_error = ?,
            pre_finished_at = CASE WHEN ? IN ('done', 'failed') THEN ? ELSE NULL END,
            status = CASE WHEN ? = 'done' THEN 'preprocessed' ELSE status END,
            updated_at = ?
        WHERE id = ?
        """,
        (pre_status, result_json, pre_error,
         pre_status, now,
         pre_status,
         now, rec_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Status updates (training)
# ---------------------------------------------------------------------------

def update_train_status(
    conn: sqlite3.Connection,
    rec_id: int,
    train_status: str,
    train_result: Optional[Dict] = None,
    train_error: Optional[str] = None,
    model_version: Optional[str] = None,
) -> None:
    """
    更新录音的训练状态。

    train_status='done' 时同时将录音 status 更新为 'trained'。
    """
    now = __import__("datetime").datetime.now().isoformat()
    result_json = json.dumps(train_result, ensure_ascii=False) if train_result else None

    conn.execute(
        """
        UPDATE recordings
        SET train_status = ?,
            train_result = ?,
            train_error = ?,
            train_finished_at = CASE WHEN ? IN ('done', 'failed') THEN ? ELSE NULL END,
            status = CASE WHEN ? = 'done' THEN 'trained' ELSE status END,
            model_version = CASE WHEN ? IS NOT NULL THEN ? ELSE model_version END,
            updated_at = ?
        WHERE id = ?
        """,
        (train_status, result_json, train_error,
         train_status, now,
         train_status,
         model_version, model_version,
         now, rec_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def count_pending_preprocess(conn: sqlite3.Connection) -> int:
    """统计待预处理的录音数。"""
    cursor = conn.execute(
        "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status = 'pending'"
    )
    return cursor.fetchone()["cnt"]


def count_pending_train(conn: sqlite3.Connection) -> int:
    """统计待训练的录音数（仅统计有预处理切段的录音）。"""
    cursor = conn.execute(
        "SELECT COUNT(*) AS cnt FROM recordings "
        "WHERE status = 'preprocessed' AND train_status = 'pending' "
        "AND JSON_TYPE(pre_result, '$.segment_count') = 'integer' "
        "AND CAST(JSON_EXTRACT(pre_result, '$.segment_count') AS INTEGER) > 0"
    )
    return cursor.fetchone()["cnt"]


def get_ready_for_training(
    conn: sqlite3.Connection,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    获取已完成预处理且尚未训练的录音清单（仅统计有预处理切段的录音）。
    通常由增量训练模块调用。

    Args:
        conn: 数据库连接。
        limit: 最大返回条数。-1 表示无限制。
    """
    if limit < 0:
        cursor = conn.execute(
            """
            SELECT * FROM recordings
            WHERE status = 'preprocessed'
              AND train_status = 'pending'
              AND JSON_TYPE(pre_result, '$.segment_count') = 'integer'
              AND CAST(JSON_EXTRACT(pre_result, '$.segment_count') AS INTEGER) > 0
            ORDER BY created_at ASC
            """,
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM recordings
            WHERE status = 'preprocessed'
              AND train_status = 'pending'
              AND JSON_TYPE(pre_result, '$.segment_count') = 'integer'
              AND CAST(JSON_EXTRACT(pre_result, '$.segment_count') AS INTEGER) > 0
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
    return [dict(r) for r in cursor.fetchall()]


def get_recordings_by_agent(
    conn: sqlite3.Connection,
    agent_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """查询指定坐席的所有录音。"""
    cursor = conn.execute(
        "SELECT * FROM recordings WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
        (agent_id, limit),
    )
    return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Model version operations
# ---------------------------------------------------------------------------

def get_active_model(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """获取当前生效的模型版本。"""
    cursor = conn.execute(
        "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def insert_model_version(
    conn: sqlite3.Connection,
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
    model_md5: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """插入模型版本记录。"""
    cursor = conn.execute(
        """
        INSERT INTO model_versions
            (version, eval_metric, eval_value, prev_eval_value, improved,
             train_recording_count, train_speaker_count, train_time_sec,
             model_path, model_md5, is_active, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (version, eval_metric, eval_value, prev_eval_value, 1 if improved else 0,
         train_recording_count, train_speaker_count, train_time_sec,
         model_path, model_md5, notes),
    )
    conn.commit()
    return cursor.lastrowid


def deactivate_model(conn: sqlite3.Connection, version: str) -> None:
    """将指定版本设为非活跃。"""
    conn.execute(
        "UPDATE model_versions SET is_active = 0 WHERE version = ?", (version,)
    )
    conn.commit()


def activate_model(conn: sqlite3.Connection, version: str) -> None:
    """将指定版本设为活跃，同时将其他版本取消活跃。"""
    conn.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
    conn.execute(
        "UPDATE model_versions SET is_active = 1 WHERE version = ?", (version,)
    )
    conn.commit()


def get_all_model_versions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """获取所有模型版本记录。"""
    cursor = conn.execute(
        "SELECT * FROM model_versions ORDER BY created_at DESC"
    )
    return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Recovery: reset stuck 'processing' records
# ---------------------------------------------------------------------------

def recover_stuck_tasks(conn: sqlite3.Connection, timeout_minutes: int = 30) -> int:
    """
    恢复卡在 'processing' 状态的录音条数。

    启动时调用，将超过 timeout_minutes 仍为 processing 的记录退回 pending。
    """
    total = 0
    for field in ("pre_status", "train_status"):
        cursor = conn.execute(
            f"""
            UPDATE recordings
            SET {field} = 'pending',
                updated_at = datetime('now')
            WHERE {field} = 'processing'
              AND updated_at < datetime('now', '-{timeout_minutes} minutes')
            """
        )
        total += cursor.rowcount
    if total:
        logger.warning("Recovered %d stuck 'processing' records", total)
    conn.commit()
    return total


# ---------------------------------------------------------------------------
# Voiceprint operations (speaker_voiceprints table)
# ---------------------------------------------------------------------------


def upsert_voiceprint(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    speaker_type: str,
    speaker_id: str,
    embedding: np.ndarray,
    segment_count: int = 1,
    source_call_ids: Optional[List[str]] = None,
) -> int:
    """注册或更新声纹。embedding 应为已 L2 归一化的 float32 ndarray。"""
    blob = embedding.astype(np.float32).tobytes()
    call_ids_json = json.dumps(source_call_ids, ensure_ascii=False) if source_call_ids else None

    # INSERT OR REPLACE: if same (model_name, speaker_type, speaker_id), replace
    conn.execute(
        """INSERT OR REPLACE INTO speaker_voiceprints
           (model_name, speaker_type, speaker_id, embedding, segment_count, source_call_ids)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_name, speaker_type, speaker_id, blob, segment_count, call_ids_json),
    )
    conn.commit()
    cursor = conn.execute(
        "SELECT id FROM speaker_voiceprints WHERE model_name=? AND speaker_type=? AND speaker_id=?",
        (model_name, speaker_type, speaker_id),
    )
    row = cursor.fetchone()
    return row["id"] if row else -1


def get_agent_voiceprint(
    conn: sqlite3.Connection,
    model_name: str = "CAM++",
) -> Optional[Dict[str, Any]]:
    """获取最近注册的坐席声纹。"""
    row = conn.execute(
        "SELECT * FROM speaker_voiceprints "
        "WHERE model_name=? AND speaker_type='agent' "
        "ORDER BY id DESC LIMIT 1",
        (model_name,),
    ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["embedding"] = np.frombuffer(result["embedding"], dtype=np.float32)
    return result


# ---------------------------------------------------------------------------
# Single-instance lock (fcntl-based, POSIX)
# ---------------------------------------------------------------------------

_FILE_LOCKS: Dict[str, int] = {}


def single_instance_lock(name: str) -> bool:
    """
    防止同一模块启动多个实例。
    使用 fcntl 文件锁，进程退出时自动释放。

    Returns:
        True 如果成功获得锁，False 如果另一个实例已在运行。
    """
    import atexit
    import fcntl

    lock_path = Path(f"/tmp/asv_train_{name}.lock")
    try:
        fd = lock_path.open("w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        logger.error("另一个 %s 进程已在运行 (lock: %s)", name, lock_path)
        return False

    _FILE_LOCKS[name] = fd

    @atexit.register
    def _release():
        if name in _FILE_LOCKS:
            try:
                fcntl.flock(_FILE_LOCKS[name], fcntl.LOCK_UN)
                _FILE_LOCKS[name].close()
            except Exception:
                pass
            del _FILE_LOCKS[name]

    return True
