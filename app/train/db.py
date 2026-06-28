"""
SQLite 数据库服务（同步版，供 CLI 端使用）。

兼容 train/ 下的预处理、训练、评分等子模块。
统一 DDL 来源：data.database.init_schema()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.database import DDL_TABLES, get_connection as unified_get_connection

logger = logging.getLogger("asv.train.db")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# __file__ is app/train/db.py → parent.parent is app/ → /data/training.db
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "training.db"


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection to the training database, creating it if needed."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema — delegate to unified data.database.init_schema()
# ---------------------------------------------------------------------------


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema using unified DDL from data.database."""
    from data.database import DDL_TABLES

    for tbl_name, ddl in DDL_TABLES.items():
        conn.execute(ddl)

    # Legacy migration: add old-style columns that existing code still needs
    # so that the mixed old/new DB schema survives until full migration.
    _ensure_legacy_columns(conn)

    conn.commit()


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    """Add legacy columns that old code paths still write to."""
    # model_versions: old monitoring/metric columns
    legacy_mv_cols = {
        "eval_metric": "TEXT",
        "eval_value": "REAL",
        "prev_eval_value": "REAL",
        "improved": "INTEGER DEFAULT 0",
        "train_recording_count": "INTEGER DEFAULT 0",
        "train_speaker_count": "INTEGER DEFAULT 0",
        "train_time_sec": "REAL",
        "model_md5": "TEXT",
        "is_active": "INTEGER DEFAULT 0",
        "notes": "TEXT",
        "previous_version": "TEXT",
        "file_size": "INTEGER",
    }
    _safe_add_columns(conn, "model_versions", legacy_mv_cols)

    # recordings: pre/train status columns used by the task-pipeline
    legacy_rec_cols = {
        "pre_status": "TEXT DEFAULT 'pending'",
        "pre_result": "TEXT",
        "pre_error": "TEXT",
        "pre_finished_at": "TEXT",
        "pre_queued_at": "TEXT",
        "train_status": "TEXT DEFAULT 'pending'",
        "train_result": "TEXT",
        "train_error": "TEXT",
        "train_finished_at": "TEXT",
        "file_hash": "TEXT",
    }
    _safe_add_columns(conn, "recordings", legacy_rec_cols)


def _safe_add_columns(
    conn: sqlite3.Connection, table: str, cols: Dict[str, str]
) -> None:
    """Add columns to a table if they don't exist."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col_name, col_type in cols.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


def claim_pending_recordings(conn: sqlite3.Connection, limit: int = 1) -> List[Dict[str, Any]]:
    """Claim recordings with pre_status='pending' for processing.

    Atomically sets pre_status='processing', returns up to `limit` rows.
    """
    rows = conn.execute(
        """SELECT id, biz_system, call_id, agent_id, customer_id,
                  call_timestamp, local_audio_path, status, pre_status
           FROM recordings
           WHERE (pre_status IS NULL OR pre_status = 'pending'
                  OR pre_status = 'failed')
           ORDER BY id
           LIMIT ?""",
        (limit,),
    ).fetchall()

    if not rows:
        return []

    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE recordings SET pre_status = 'processing', pre_queued_at = NULL "
        f"WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    return [dict(r) for r in rows]


def update_pre_status(
    conn: sqlite3.Connection,
    rec_id: int,
    status: str,
    *,
    pre_result: Any = None,
    pre_error: Optional[str] = None,
) -> None:
    """Update the preprocess status for a recording."""
    now = datetime.now().isoformat()
    # done / unsegmentable 都表示 VAD 流程正常完成，记 pre_result + pre_finished_at
    # unsegmentable: VAD 跑完但无语音段（静音/纯噪声），仍属"预处理完成"
    if status in ("done", "unsegmentable"):
        conn.execute(
            """UPDATE recordings
               SET pre_status=?, pre_result=?, pre_finished_at=?, pre_queued_at=NULL,
                   status='preprocessed'
               WHERE id=?""",
            (status, json.dumps(pre_result) if pre_result else None, now, rec_id),
        )
    elif status == "failed":
        conn.execute(
            """UPDATE recordings
               SET pre_status=?, pre_result=?, pre_error=?, pre_finished_at=?, pre_queued_at=NULL
               WHERE id=?""",
            (status, json.dumps(pre_result) if pre_result else None, pre_error, now, rec_id),
        )
    else:
        conn.execute(
            "UPDATE recordings SET pre_status=?, pre_queued_at=NULL WHERE id=?",
            (status, rec_id),
        )
    conn.commit()


def count_pending_preprocess(conn: sqlite3.Connection) -> int:
    """Count recordings waiting for preprocessing."""
    cur = conn.execute(
        "SELECT COUNT(*) AS cnt FROM recordings WHERE pre_status='pending' OR pre_status='failed'"
    )
    row = cur.fetchone()
    return row["cnt"] if row else 0


def get_recording_by_id(conn: sqlite3.Connection, rec_id: int) -> Optional[Dict[str, Any]]:
    """Look up a recording by its primary key id."""
    row = conn.execute(
        "SELECT * FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    return dict(row) if row else None


def count_pending_train(conn: sqlite3.Connection) -> int:
    """Count recordings waiting for training."""
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM recordings WHERE train_status='pending'"
    ).fetchone()
    return row["cnt"] if row else 0


def get_ready_for_training(
    conn: sqlite3.Connection, limit: int = -1
) -> List[Dict[str, Any]]:
    """Get recordings that are ready for training (preprocessed but not yet trained)."""
    if limit > 0:
        rows = conn.execute(
            """SELECT id, biz_system, call_id, agent_id, customer_id,
                      call_timestamp, local_audio_path, status, pre_status,
                      train_status
               FROM recordings
               WHERE train_status='pending' AND pre_status='done'
               LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, biz_system, call_id, agent_id, customer_id,
                      call_timestamp, local_audio_path, status, pre_status,
                      train_status
               FROM recordings
               WHERE train_status='pending' AND pre_status='done'"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_train_status(
    conn: sqlite3.Connection,
    rec_id: int,
    status: str,
    *,
    train_result: Any = None,
    train_error: Optional[str] = None,
    model_version: Optional[str] = None,
) -> None:
    """Update training status for a recording."""
    now = datetime.now().isoformat()
    conn.execute(
        """UPDATE recordings
           SET train_status=?, train_result=?, train_error=?,
               train_finished_at=?, model_version=?
           WHERE id=?""",
        (status,
         json.dumps(train_result) if train_result else None,
         train_error, now, model_version, rec_id),
    )
    conn.commit()


def recover_stuck_tasks(conn: sqlite3.Connection) -> int:
    """Reset any stuck 'processing' recordings back to 'pending'.

    记录 pre_queued_at 为恢复时间，界面据此提示"异常恢复，建议重试"。
    """
    now = datetime.now().isoformat()
    count = conn.execute(
        "UPDATE recordings SET pre_status='pending', pre_queued_at=? "
        "WHERE pre_status='processing'",
        (now,),
    ).rowcount
    count += conn.execute(
        "UPDATE recordings SET train_status='pending' WHERE train_status='training'"
    ).rowcount
    if count:
        conn.commit()
    return count


def single_instance_lock(name: str, timeout: int = 30) -> bool:
    """Simple single-instance lock using a lock file.
    
    If the lock cannot be acquired immediately, retries every 5s
    up to `timeout` seconds before giving up.
    """
    lock_path = Path(f"/tmp/asv_{name}.lock")
    try:
        import fcntl
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        lock_path.write_text(str(os.getpid()))
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (IOError, OSError):
                if time.monotonic() >= deadline:
                    logger.error("等待 %s 锁超时 (%ds)，放弃", name, timeout)
                    return False
                logger.info("等待 %s 锁释放… (pid=%s)", name, lock_path.read_text().strip())
                time.sleep(5)
    except (ImportError, OSError) as e:
        logger.warning("锁机制不可用 (%s)，跳过", e)
        return True


# ---------------------------------------------------------------------------
# Model versions — write to both old & new columns for backward compat
# ---------------------------------------------------------------------------


def get_active_model(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Return the currently published (active) model version, if any.

    Uses the unified 'status' column: status='published' AND
    (old-style is_active=1 OR no old column).
    """
    try:
        row = conn.execute(
            """SELECT * FROM model_versions
               WHERE status = 'published'
               ORDER BY id DESC LIMIT 1"""
        ).fetchone()
    except Exception:
        row = None
    if row:
        return dict(row)
    # Fallback: old-style is_active
    try:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except Exception:
        return None
    return dict(row) if row else None


def insert_model_version(
    conn: sqlite3.Connection,
    *,
    model_name: str = "Unknown",
    version_tag: str = "",
    version: str,
    embedding_dim: int = 192,
    eval_metric: str = "EER",
    eval_value: float = 0.0,
    prev_eval_value: Optional[float] = None,
    improved: bool = False,
    train_recording_count: int = 0,
    train_speaker_count: int = 0,
    train_time_sec: float = 0.0,
    model_path: str = "",
    model_md5: Optional[str] = None,
    base_model: str = "",
    config: str = "{}",
    metrics: Optional[str] = None,
    notes: Optional[str] = None,
    is_active: bool = True,
    score: Optional[float] = None,
) -> int:
    """Insert a model version record, writing both unified & legacy columns.

    Returns the new row id.
    """
    if score is None:
        score = eval_value
    if not version_tag:
        version_tag = version

    if metrics is not None:
        metrics_str = metrics
    else:
        metrics_str = json.dumps({
            "eval_metric": eval_metric,
            "eval_value": eval_value,
            "prev_eval_value": prev_eval_value,
            "improved": improved,
            "train_recording_count": train_recording_count,
            "train_speaker_count": train_speaker_count,
            "train_time_sec": train_time_sec,
            "model_md5": model_md5,
            "notes": notes,
            "previous_version": None,
        }, ensure_ascii=False)

    conn.execute(
        """INSERT INTO model_versions
            (model_name, version, version_tag, embedding_dim, base_model,
             config, metrics, score, model_path, file_size, status,
             eval_metric, eval_value, prev_eval_value, improved,
             train_recording_count, train_speaker_count, train_time_sec,
             model_md5, is_active, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'published',
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_name, version, version_tag, embedding_dim, base_model,
            config, metrics_str, score, model_path,
            0,  # file_size placeholder
            eval_metric, eval_value, prev_eval_value, 1 if improved else 0,
            train_recording_count, train_speaker_count, train_time_sec,
            model_md5, 1 if is_active else 0, notes,
        ),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def deactivate_model(conn: sqlite3.Connection, version: str) -> None:
    """Deactivate a model version (both old and new columns)."""
    conn.execute("UPDATE model_versions SET status='archived' WHERE version=?", (version,))
    conn.execute("UPDATE model_versions SET is_active=0 WHERE version=?", (version,))
    conn.commit()


def activate_model(conn: sqlite3.Connection, version: str) -> None:
    """Activate a model version (both old and new columns)."""
    conn.execute("UPDATE model_versions SET status='published' WHERE version=?", (version,))
    conn.execute("UPDATE model_versions SET is_active=1 WHERE version=?", (version,))
    # Ensure only one active
    conn.execute(
        """UPDATE model_versions SET is_active=0, status='archived'
           WHERE version != ? AND (is_active=1 OR status='published')""",
        (version,),
    )
    conn.commit()


def get_all_model_versions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM model_versions ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Speaker voiceprints
# ---------------------------------------------------------------------------


def get_speaker_voiceprints(
    conn: sqlite3.Connection,
    model_name: str,
    speaker_type: str = "agent",
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """SELECT id, model_name, speaker_type, speaker_id, embedding,
                  segment_count, source_call_ids, created_at
           FROM speaker_voiceprints
           WHERE model_name=? AND speaker_type=?
           ORDER BY speaker_id""",
        (model_name, speaker_type),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_speaker_voiceprint(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    speaker_type: str,
    speaker_id: str,
    embedding: bytes,
    segment_count: int = 1,
    source_call_ids: str = "",
) -> int:
    conn.execute(
        """INSERT INTO speaker_voiceprints
            (model_name, speaker_type, speaker_id, embedding,
             segment_count, source_call_ids)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_name, speaker_type, speaker_id, embedding,
         segment_count, source_call_ids),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def upsert_voiceprint(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    speaker_type: str,
    speaker_id: str,
    embedding: bytes,
    segment_count: int = 1,
    source_call_ids: str = "",
) -> int:
    """Insert or update a speaker voiceprint keyed by (model_name, speaker_type, speaker_id).

    If a row already exists for the same triple, the embedding and metadata
    are overwritten and the segment_count is accumulated.
    """
    existing = conn.execute(
        "SELECT id, segment_count FROM speaker_voiceprints "
        "WHERE model_name=? AND speaker_type=? AND speaker_id=? LIMIT 1",
        (model_name, speaker_type, speaker_id),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE speaker_voiceprints "
            "SET embedding=?, segment_count=segment_count+?, source_call_ids=? "
            "WHERE id=?",
            (embedding, segment_count, source_call_ids, existing[0]),
        )
        conn.commit()
        return existing[0]
    else:
        return insert_speaker_voiceprint(
            conn,
            model_name=model_name,
            speaker_type=speaker_type,
            speaker_id=speaker_id,
            embedding=embedding,
            segment_count=segment_count,
            source_call_ids=source_call_ids,
        )
