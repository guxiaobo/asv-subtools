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

from services.recording_db import insert_recording, get_recording
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


def _download_from_url(url: str, save_dir: Path) -> str:
    """从 URL 下载录音文件到本地。"""
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    save_path = save_dir / f"{url_hash}.wav"

    if save_path.exists():
        logger.debug("URL cache HIT: %s -> %s", url, save_path)
        return str(save_path)

    logger.info("Downloading audio from URL: %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ASV-API/1.0 (RecordingReceiver)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            audio_bytes = resp.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download audio from URL: {e}",
        )

    save_path.write_bytes(audio_bytes)
    logger.info("URL download saved: %s -> %s (%d bytes)", url, save_path, len(audio_bytes))
    return str(save_path)


def _fetch_by_id(audio_id: str, save_dir: Path) -> str:
    """
    通过录音 ID 从 fetcher 插件拉取。
    对于 'local_file' fetcher，搜索 NAS mount 路径下的文件。
    """
    from services.fetcher import AudioFetcher, FetchError

    fetcher_type = app_config.fetcher_type
    try:
        fetcher = AudioFetcher.create(fetcher_type)
    except ValueError:
        fetcher = AudioFetcher.create("local_file")
    try:
        audio_bytes = fetcher.fetch(audio_id)
    except FetchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch audio by ID '{audio_id}': {e}",
        )

    save_path = save_dir / f"{audio_id}.wav"
    save_path.write_bytes(audio_bytes)
    logger.info("Fetched by ID '%s' -> %s (%d bytes)", audio_id, save_path, len(audio_bytes))
    return str(save_path)


# ---------------------------------------------------------------------------
# POST /api/v1/recordings/push
# ---------------------------------------------------------------------------

@router.post("/recordings/push", response_model=dict)
async def push_recording(
    # Metadata (form fields)
    biz_system: str = Form(..., description="业务系统: collection | cs"),
    agent_id: str = Form(..., min_length=1, max_length=64, description="坐席工号"),
    customer_phone: str = Form(..., min_length=1, max_length=32, description="客户脱敏号码"),
    call_timestamp: str = Form(..., description="通话时间 (ISO 8601)"),
    call_id: str = Form(..., min_length=1, max_length=128, description="通话唯一 ID"),
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

    流程：
      接收元数据 + 录音 → 保存到本地录音目录 → 写入 SQLite 数据库
    """
    start = time.time()

    # Validate biz_system
    if biz_system not in ("collection", "cs"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"biz_system 必须是 'collection' 或 'cs', 收到: '{biz_system}'",
        )

    # Validate audio_source_type
    if audio_source_type not in ("binary", "url", "id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"audio_source_type 必须是 'binary' | 'url' | 'id', 收到: '{audio_source_type}'",
        )

    # Resolve save directory
    recordings_root = _get_recordings_root()
    date_part = call_timestamp[:10] if len(call_timestamp) >= 10 else "unknown"
    save_dir = recordings_root / biz_system / date_part
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve audio source (async, non-blocking) --------------------------
    local_path: Optional[str] = None
    audio_original_url: Optional[str] = None

    if audio_source_type == "binary":
        if audio_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='binary' 但未提供 audio_data 文件",
            )
        local_path = await asyncio.to_thread(_save_upload_file, audio_data, save_dir)

    elif audio_source_type == "url":
        if not audio_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='url' 但未提供 audio_url",
            )
        audio_original_url = audio_url
        local_path = await asyncio.to_thread(_download_from_url, audio_url, save_dir)

    elif audio_source_type == "id":
        if not audio_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_source_type='id' 但未提供 audio_id",
            )
        local_path = await asyncio.to_thread(_fetch_by_id, audio_id, save_dir)

    # --- Write to SQLite (独立连接，无锁) ------------------------------------
    _db_path = app_config.training.db_path or None
    rec_id = await insert_recording(
        biz_system=biz_system,
        call_id=call_id,
        agent_id=agent_id,
        customer_phone=customer_phone,
        call_timestamp=call_timestamp,
        audio_source_type=audio_source_type,
        local_audio_path=local_path,
        audio_original_url=audio_original_url,
        channel_separated=channel_separated,
        duration_sec=duration_sec,
        db_path=_db_path,
    )

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "Recording pushed: id=%d call_id=%s %s/%s (%dms)",
        rec_id, call_id, biz_system, agent_id, elapsed_ms,
    )

    return JSONResponse(content={
        "success": True,
        "data": {
            "recording_id": rec_id,
            "call_id": call_id,
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
