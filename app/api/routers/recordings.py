"""
录音推送 API 路由。

POST /api/v1/recordings/push
  供催收和客服业务系统实时推送通话录音信息。

支持三种音频来源：
  - binary:  multipart/form-data 二进制上传
  - url:     从 URL 下载录音文件
  - id:      通过 fetcher 插件从业务系统拉取
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.recording_db import insert_recording, get_recording, get_recording_by_hash
from config import app_config

logger = logging.getLogger("asv-api.recordings")

router = APIRouter(prefix="/api/v1", tags=["recordings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_recordings_root() -> Path:
    """获取录音保存根目录。"""
    root_path = app_config.training.recordings_root
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_upload_file(upload: UploadFile, save_dir: Path) -> str:
    """保存上传的录音文件到本地，保留原始扩展名。（同步，被 asyncio.to_thread 调用）"""
    original_name = upload.filename or "recording"
    save_path = save_dir / original_name
    content = upload.file.read()
    save_path.write_bytes(content)
    logger.info("Saved upload '%s' -> %s (%d bytes)", upload.filename, save_path, len(content))
    return str(save_path)


def _download_to_bytes(url: str) -> bytes:
    """从 URL 下载音频字节流（不写磁盘）。"""
    logger.info("Downloading audio from URL: %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ASV-API/1.0 (RecordingReceiver)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download audio from URL: {e}",
        )


def _fetch_to_bytes(audio_id: str) -> bytes:
    """通过录音 ID 从 fetcher 插件拉取音频字节流（不写磁盘）。"""
    from services.fetcher import AudioFetcher, FetchError

    fetcher_type = app_config.fetcher_type
    try:
        fetcher = AudioFetcher.create(fetcher_type)
    except ValueError:
        fetcher = AudioFetcher.create("local_file")
    try:
        return fetcher.fetch(audio_id)
    except FetchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch audio by ID '{audio_id}': {e}",
        )


def _save_bytes_to_disk(data: bytes, filename: str, save_dir: Path) -> str:
    """将字节流写入磁盘。返回绝对路径。"""
    save_path = Path(save_dir) / filename
    save_path.write_bytes(data)
    logger.info("Saved '%s' -> %s (%d bytes)", filename, save_path, len(data))
    return str(save_path)


# ---------------------------------------------------------------------------
# POST /api/v1/recordings/push
# ---------------------------------------------------------------------------

def _extract_customer_id_from_filename(filename: str) -> str:
    """从录音文件名提取客户 ID：第一个 '-' 之前的字符串。

    示例：
      张三-2606071621.mp3       → '张三'
      18148506696-2606080924.mp3 → '18148506696'
      张三-2606071621-1.mp3      → '张三'
    """
    if not filename:
        return ""
    name_no_ext = filename.rsplit(".", 1)[0]
    idx = name_no_ext.find("-")
    if idx > 0:
        return name_no_ext[:idx]
    return name_no_ext


def _extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """从录音文件名提取时间戳并转为 ISO 8601。

    文件名格式：{客户}-{YYMMDDHHMM}[-N].mp3
    YYMMDDHHMM 是 10 位数字时间戳。

    示例：
      张三-2606071621.mp3       → '2026-06-07T16:21:00'
      张三-2606071621-1.mp3     → '2026-06-07T16:21:00'
      18148506696-2606080924.mp3 → '2026-06-08T09:24:00'
    """
    if not filename:
        return None
    name_no_ext = filename.rsplit(".", 1)[0]
    # 取第一个 '-' 后的内容
    idx = name_no_ext.find("-")
    if idx < 0:
        return None
    after_first_dash = name_no_ext[idx + 1:]
    # 取前 10 位数字作为 YYMMDDHHMM
    ts_candidate = ""
    for ch in after_first_dash:
        if ch.isdigit() and len(ts_candidate) < 10:
            ts_candidate += ch
        elif len(ts_candidate) == 10:
            break
    if len(ts_candidate) != 10:
        return None
    try:
        YY = ts_candidate[0:2]
        MM = ts_candidate[2:4]
        DD = ts_candidate[4:6]
        HH = ts_candidate[6:8]
        MI = ts_candidate[8:10]
        return f"20{YY}-{MM}-{DD}T{HH}:{MI}:00"
    except Exception:
        return None


@router.post("/recordings/push", response_model=dict)
async def push_recording(
    # Metadata (form fields)
    biz_system: str = Form(..., description="业务系统: collection | cs"),
    agent_id: str = Form(..., min_length=1, max_length=64, description="坐席工号"),
    customer_phone: Optional[str] = Form(default=None, description="客户脱敏号码（可选，未提供则从文件名自动提取）"),
    call_timestamp: Optional[str] = Form(default=None, description="通话时间 (ISO 8601)（可选，未提供则从文件名自动提取 YYMMDDHHMM）"),
    call_id: Optional[str] = Form(default=None, description="通话唯一 ID（可选，未提供则使用文件名不含扩展名）"),
    audio_source_type: str = Form(..., description="音频来源: binary | url | id"),
    # Audio file (binary mode)
    audio_data: Optional[UploadFile] = File(default=None, description="二进制音频文件"),
    # URL mode
    audio_url: Optional[str] = Form(default=None, description="录音文件 URL"),
    # ID mode
    audio_id: Optional[str] = Form(default=None, description="业务系统录音 ID"),
    # Optional metadata
    channel_separated: bool = Form(default=False, description="是否已通道分离"),
    duration_sec: Optional[float] = Form(default=None, description="通话时长(秒)"),
    # Fetcher backend (for ID mode)
    storage_backend: Optional[str] = Form(default="local_file"),
    bucket: Optional[str] = Form(default=None),
) -> JSONResponse:
    """
    接收业务系统推送的通话录音。

    三种音频来源：
    1. **binary**: 通过 multipart `audio_data` 字段直接上传音频文件
    2. **url**:    提供录音文件的 HTTP/HTTPS 可下载 URL
    3. **id**:     提供业务系统内部录音 ID，由 fetcher 插件拉取

    客户 ID 规则（binary 模式）：
      优先使用传入的 customer_phone，若未提供则自动从音频文件名提取
      （第一个 "-" 之前的字符串作为客户 ID）。

    流程：
      接收元数据 + 录音 → 保存到本地录音目录 → 写入 SQLite 数据库
    """
    start = time.time()

    # ── 客户 ID / 时间戳 / call_id 自动提取（binary 模式） ──
    resolved_customer_phone = customer_phone or ""
    resolved_call_timestamp = call_timestamp
    resolved_call_id = call_id

    if audio_source_type == "binary":
        if audio_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='binary' 但未提供 audio_data 文件",
            )
        filename = audio_data.filename or ""
        # 客户 ID：传参优先，否则从文件名提取
        if not resolved_customer_phone and filename:
            resolved_customer_phone = _extract_customer_id_from_filename(filename)
            logger.info(
                "客户 ID 从文件名自动提取: '%s' → '%s'",
                filename, resolved_customer_phone,
            )
        # 时间戳：传参优先，否则从文件名提取 YYMMDDHHMM
        if not resolved_call_timestamp and filename:
            resolved_call_timestamp = (
                _extract_timestamp_from_filename(filename)
                or resolved_call_timestamp
            )
            if resolved_call_timestamp:
                logger.info(
                    "时间戳从文件名自动提取: '%s' → '%s'",
                    filename, resolved_call_timestamp,
                )
        # call_id：传参优先，否则用文件名（不含扩展名）
        if not resolved_call_id and filename:
            resolved_call_id = filename.rsplit(".", 1)[0]
            logger.info(
                "call_id 从文件名自动提取: '%s' → '%s'",
                filename, resolved_call_id,
            )

    # 最终校验
    if not resolved_call_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="call_timestamp 未提供且无法从文件名自动提取",
        )
    if not resolved_call_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="call_id 未提供且无法从文件名自动提取",
        )

    # 合法性校验：biz_system
    if biz_system not in ("collection", "cs"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"biz_system 必须是 'collection' 或 'cs', 收到: '{biz_system}'",
        )

    # 合法性校验：audio_source_type
    if audio_source_type not in ("binary", "url", "id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"audio_source_type 必须是 'binary' | 'url' | 'id', 收到: '{audio_source_type}'",
        )

    # Resolve save directory
    recordings_root = _get_recordings_root()
    date_part = resolved_call_timestamp[:10] if len(resolved_call_timestamp) >= 10 else "unknown"
    save_dir = recordings_root / biz_system / date_part
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve audio source + MD5 hash (async, non-blocking) -----------------
    local_path: Optional[str] = None
    audio_original_url: Optional[str] = None
    file_hash: Optional[str] = None

    if audio_source_type == "binary":
        # audio_data is guaranteed not-None by the check above
        assert audio_data is not None  # type guard for LSP
        # Read file content once
        content = await audio_data.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的音频文件为空",
            )
        # Calculate MD5 hash
        file_hash = hashlib.md5(content).hexdigest()
        logger.debug("Binary upload '%s' MD5: %s", audio_data.filename or "?", file_hash)

        # Check duplicate by hash (before saving to disk)
        _db_path = app_config.training.db_path or None
        existing = await get_recording_by_hash(file_hash, db_path=_db_path)
        if existing:
            logger.info(
                "重复录音 (hash=%s): '%s' → 返回已有记录 id=%d call_id=%s",
                file_hash, audio_data.filename or "?",
                existing["id"], existing["call_id"],
            )
            return JSONResponse(content={
                "success": True,
                "duplicate": True,
                "data": {
                    "recording_id": existing["id"],
                    "call_id": existing["call_id"],
                    "customer_phone": existing["customer_phone"],
                    "call_timestamp": existing["call_timestamp"],
                    "local_path": existing["local_audio_path"] or "",
                    "status": existing["status"],
                },
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            })

        # New file — save to disk
        filename = audio_data.filename or "recording"
        save_path = save_dir / filename
        await asyncio.to_thread(save_path.write_bytes, content)
        local_path = str(save_path)
        logger.info("Saved '%s' -> %s (%d bytes)", filename, save_path, len(content))

    elif audio_source_type == "url":
        if not audio_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='url' 但未提供 audio_url",
            )
        audio_original_url = audio_url
        content = await asyncio.to_thread(_download_to_bytes, audio_url)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="从 URL 下载的音频为空",
            )
        file_hash = hashlib.md5(content).hexdigest()
        logger.debug("URL download '%s' MD5: %s", audio_url, file_hash)

        # Check duplicate by hash (before saving to disk)
        _db_path = app_config.training.db_path or None
        existing = await get_recording_by_hash(file_hash, db_path=_db_path)
        if existing:
            logger.info(
                "重复录音 (hash=%s): url='%s' → 返回已有记录 id=%d call_id=%s",
                file_hash, audio_url, existing["id"], existing["call_id"],
            )
            return JSONResponse(content={
                "success": True,
                "duplicate": True,
                "data": {
                    "recording_id": existing["id"],
                    "call_id": existing["call_id"],
                    "customer_phone": existing["customer_phone"],
                    "call_timestamp": existing["call_timestamp"],
                    "local_path": existing["local_audio_path"] or "",
                    "status": existing["status"],
                },
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            })

        # New file — save to disk
        url_filename = f"{hashlib.md5(audio_url.encode('utf-8')).hexdigest()}.wav"
        local_path = await asyncio.to_thread(
            _save_bytes_to_disk, content, url_filename, save_dir
        )

    elif audio_source_type == "id":
        if not audio_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='id' 但未提供 audio_id",
            )
        content = await asyncio.to_thread(_fetch_to_bytes, audio_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"通过 ID '{audio_id}' 获取的音频为空",
            )
        file_hash = hashlib.md5(content).hexdigest()
        logger.debug("Fetch by ID '%s' MD5: %s", audio_id, file_hash)

        # Check duplicate by hash (before saving to disk)
        _db_path = app_config.training.db_path or None
        existing = await get_recording_by_hash(file_hash, db_path=_db_path)
        if existing:
            logger.info(
                "重复录音 (hash=%s): audio_id='%s' → 返回已有记录 id=%d call_id=%s",
                file_hash, audio_id, existing["id"], existing["call_id"],
            )
            return JSONResponse(content={
                "success": True,
                "duplicate": True,
                "data": {
                    "recording_id": existing["id"],
                    "call_id": existing["call_id"],
                    "customer_phone": existing["customer_phone"],
                    "call_timestamp": existing["call_timestamp"],
                    "local_path": existing["local_audio_path"] or "",
                    "status": existing["status"],
                },
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            })

        # New file — save to disk
        id_filename = f"{audio_id}.wav"
        local_path = await asyncio.to_thread(
            _save_bytes_to_disk, content, id_filename, save_dir
        )

    # --- Write to SQLite (独立连接，无锁) ------------------------------------
    _db_path = app_config.training.db_path or None
    rec_id = await insert_recording(
        biz_system=biz_system,
        call_id=resolved_call_id,
        agent_id=agent_id,
        customer_phone=resolved_customer_phone,
        call_timestamp=resolved_call_timestamp,
        audio_source_type=audio_source_type,
        local_audio_path=local_path,
        audio_original_url=audio_original_url,
        file_hash=file_hash,
        channel_separated=channel_separated,
        duration_sec=duration_sec,
        db_path=_db_path,
    )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "Recording pushed: id=%d call_id=%s customer=%s ts=%s %s/%s (%dms)",
        rec_id, resolved_call_id, resolved_customer_phone,
        resolved_call_timestamp, biz_system, agent_id, elapsed_ms,
    )

    return JSONResponse(content={
        "success": True,
        "data": {
            "recording_id": rec_id,
            "call_id": resolved_call_id,
            "customer_phone": resolved_customer_phone,
            "call_timestamp": resolved_call_timestamp,
            "local_path": local_path or "",
            "status": "raw",
        },
        "processing_time_ms": round(elapsed_ms, 1),
    })


# ---------------------------------------------------------------------------
# GET /api/v1/recordings/stats — 统计信息
# ---------------------------------------------------------------------------

@router.get("/recordings/stats", tags=["recordings"])
async def get_recording_stats() -> JSONResponse:
    """获取录音统计信息。"""
    from services.recording_db import count_pending_preprocess, count_pending_train

    _db_path = app_config.training.db_path or None
    pending_pre = await count_pending_preprocess(db_path=_db_path)
    pending_train = await count_pending_train(db_path=_db_path)

    return JSONResponse(content={
        "success": True,
        "data": {
            "pending_preprocess": pending_pre,
            "pending_train": pending_train,
        },
    })
