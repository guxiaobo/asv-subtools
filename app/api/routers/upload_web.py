"""
坐席录音上传 Web 界面路由。

提供：
  GET  /upload          — 渲染上传页面（HTML）
  POST /upload          — 处理录音文件上传
  GET  /upload/history  — 返回最近上传记录（JSON，供页面刷新）

功能：
  - 坐席 ID、业务系统、录音文件表单
  - 后台按 biz_system/日期 目录组织保存
  - 写入 SQLite recordings 表，记录录音状态
  - 重复文件（MD5 hash）自动去重

文件名格式（两种）：
  1) {客户标识}-{YYMMDDHHMM}.ext       → 坐席 ID 从表单获取
  2) {坐席ID}-{客户标识}-{YYMMDDHHMM}.ext  → 坐席 ID 从文件名自动提取
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import app_config
from services.recording_db import insert_recording, get_recording_by_hash

logger = logging.getLogger("asv-api.upload_web")

router = APIRouter(tags=["upload"])

# ── Jinja2 templates ────────────────────────────────────────────────
_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_recordings_root() -> Path:
    """获取录音保存根目录。"""
    root_path = app_config.training.recordings_root
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    return root


# -------------------------------------------------------------------
# Validators & metadata extraction
# -------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ulaw", ".alaw"}
MIN_FILE_SIZE = 1024  # 1KB


def _parse_filename(filename: str) -> dict:
    """
    解析文件名，返回结构化信息。

    支持两种格式：
      1) {customer}-{YYMMDDHHMM}.ext              → 2 parts
      2) {agent}-{customer}-{YYMMDDHHMM}.ext       → 3 parts

    Returns:
        {
            "valid": bool,
            "error": str | None,
            "agent_id_from_filename": str,     # 空串表示未从文件名提取
            "customer_phone": str,
            "call_timestamp": str,             # ISO 8601 或空串
            "call_id": str,                     # 文件名不含扩展名
            "num_parts": int,                   # 2 或 3
        }
    """
    result = {
        "valid": False,
        "error": None,
        "agent_id_from_filename": "",
        "customer_phone": "",
        "call_timestamp": "",
        "call_id": "",
        "num_parts": 0,
    }

    if not filename:
        result["error"] = "文件名为空"
        return result

    # 检查扩展名
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        result["error"] = f"不支持的文件格式 '{ext}'，允许: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        return result

    name_no_ext = filename.rsplit(".", 1)[0]
    parts = name_no_ext.split("-")
    num_parts = len(parts)
    result["num_parts"] = num_parts
    result["call_id"] = name_no_ext

    if num_parts < 2 or num_parts > 3:
        result["error"] = (
            f"文件名格式无效：需要 2 部分 {{客户}}-{{时间戳}} 或 "
            f"3 部分 {{坐席ID}}-{{客户}}-{{时间戳}}，当前为 {num_parts} 部分"
        )
        return result

    # 时间戳总是在最后一部分
    ts_part = parts[-1]
    ts_digits = ""
    for ch in ts_part:
        if ch.isdigit() and len(ts_digits) < 10:
            ts_digits += ch

    if len(ts_digits) != 10:
        result["error"] = f"未找到有效时间戳（需要 10 位数字 YYMMDDHHMM），最后一部分: '{ts_part}'"
        return result

    # 校验时间戳合理性
    try:
        hh = int(ts_digits[6:8])
        mm = int(ts_digits[8:10])
        if hh > 23:
            result["error"] = f"时间戳小时字段无效: {ts_digits[6:8]}（应为 00-23）"
            return result
        if mm > 59:
            result["error"] = f"时间戳分钟字段无效: {ts_digits[8:10]}（应为 00-59）"
            return result
    except ValueError:
        result["error"] = "时间戳包含非数字字符"
        return result

    # 格式化时间戳
    YY = ts_digits[0:2]
    MM = ts_digits[2:4]
    DD = ts_digits[4:6]
    HH = ts_digits[6:8]
    MI = ts_digits[8:10]
    result["call_timestamp"] = f"20{YY}-{MM}-{DD}T{HH}:{MI}:00"

    # 按格式解析各字段
    if num_parts == 2:
        # {customer}-{YYMMDDHHMM}
        result["customer_phone"] = parts[0]
        if not parts[0]:
            result["error"] = "客户标识为空（'-' 前无内容）"
            return result
    else:
        # {agent_id}-{customer}-{YYMMDDHHMM}
        result["agent_id_from_filename"] = parts[0]
        result["customer_phone"] = parts[1]
        if not parts[0]:
            result["error"] = "坐席 ID 为空（第一个 '-' 前无内容）"
            return result
        if not parts[1]:
            result["error"] = "客户标识为空（第二个 '-' 前无内容）"
            return result

    result["valid"] = True
    return result


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@router.get("/upload", response_class=HTMLResponse, include_in_schema=False)
async def upload_page(request: Request) -> HTMLResponse:
    """渲染录音上传页面。"""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request},
    )


@router.post("/upload")
async def upload_recording(
    request: Request,
    agent_id: str = Form(default="", description="坐席工号（3 部分文件名时可选）"),
    biz_system: str = Form(..., description="业务系统: collection | cs"),
    audio_data: UploadFile = File(..., description="录音文件"),
) -> JSONResponse:
    """
    接收坐席前端上传的录音文件。

    流程：
      1. 解析文件名 → 获取各字段
      2. 如文件名含坐席 ID（3 部分格式），自动提取
      3. 校验业务系统、文件大小、扩展名
      4. 坐席 ID 最终确认（来自文件名或表单）
      5. 计算 MD5 hash → 查重
      6. 保存到 recordings_root/{biz_system}/{date}/
      7. 写入 SQLite recordings 表
    """
    start = time.time()

    # ── 1. 解析文件名 ──────────────────────────────────────────────
    filename = audio_data.filename or ""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未选择文件或文件名为空",
        )

    parsed = _parse_filename(filename)
    if not parsed["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=parsed["error"],
        )

    # ── 2. 确定坐席 ID ──────────────────────────────────────────────
    # 3 部分文件名 → 从文件名提取；2 部分文件名 → 从表单获取
    agent_id_from_filename = parsed.get("agent_id_from_filename", "")
    if agent_id_from_filename:
        resolved_agent_id = agent_id_from_filename.strip()
    else:
        resolved_agent_id = agent_id.strip()
        if not resolved_agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="坐席工号不能为空（2 部分文件名格式需要手动填写）",
            )

    # ── 3. 校验业务系统 ─────────────────────────────────────────────
    if biz_system not in ("collection", "cs"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"业务系统必须是 'collection' 或 'cs'，收到: '{biz_system}'",
        )

    # ── 4. 文件大小校验 ─────────────────────────────────────────────
    content = await audio_data.read()
    file_size = len(content)
    if file_size < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件过小 ({file_size} bytes)，需大于 {MIN_FILE_SIZE} bytes (1KB)",
        )

    # ── 5. 从解析结果提取元数据 ────────────────────────────────────
    customer_phone = parsed["customer_phone"]
    call_timestamp = parsed["call_timestamp"]
    call_id = parsed["call_id"]

    if not call_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法从文件名解析时间戳",
        )

    # ── 6. 计算 MD5 → 查重 ─────────────────────────────────────────
    file_hash = hashlib.md5(content).hexdigest()
    _db_path = app_config.training.db_path or None
    existing = await get_recording_by_hash(file_hash, db_path=_db_path)
    if existing:
        elapsed_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "网页上传重复录音 (hash=%s): %s (agent=%s) → 返回已有记录 id=%d",
            file_hash, filename, resolved_agent_id, existing["id"],
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
            "processing_time_ms": elapsed_ms,
        })

    # ── 7. 保存到磁盘 ───────────────────────────────────────────────
    recordings_root = _get_recordings_root()
    date_part = call_timestamp[:10] if len(call_timestamp) >= 10 else "unknown"
    save_dir = recordings_root / biz_system / date_part
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    if save_path.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        save_path = save_dir / f"{stem}_{int(time.time())}{suffix}"

    save_path.write_bytes(content)
    local_path = str(save_path)
    logger.info("网页上传保存: '%s' -> %s (%d bytes)", filename, save_path, file_size)

    # ── 8. 写入 SQLite ──────────────────────────────────────────────
    rec_id = await insert_recording(
        biz_system=biz_system,
        call_id=call_id,
        agent_id=resolved_agent_id,
        customer_phone=customer_phone,
        call_timestamp=call_timestamp,
        audio_source_type="binary",
        local_audio_path=local_path,
        file_hash=file_hash,
        channel_separated=False,
        db_path=_db_path,
    )

    elapsed_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "网页上传完成: id=%d call_id=%s agent=%s customer=%s ts=%s (%dms)",
        rec_id, call_id, resolved_agent_id, customer_phone, call_timestamp, elapsed_ms,
    )

    return JSONResponse(content={
        "success": True,
        "duplicate": False,
        "data": {
            "recording_id": rec_id,
            "call_id": call_id,
            "agent_id": resolved_agent_id,
            "customer_phone": customer_phone,
            "call_timestamp": call_timestamp,
            "local_path": local_path,
            "status": "raw",
        },
        "processing_time_ms": elapsed_ms,
    })


@router.get("/upload/history", include_in_schema=False)
async def upload_history(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100),
) -> JSONResponse:
    """返回最近上传记录（JSON 格式）。"""
    import aiosqlite

    _db_path = app_config.training.db_path or None
    path = Path(_db_path) if _db_path else (
        Path(__file__).resolve().parent.parent.parent / "data" / "training.db"
    )

    items: List[Dict[str, Any]] = []

    if not path.exists():
        return JSONResponse(content={"success": True, "data": items})

    try:
        async with aiosqlite.connect(str(path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT id, call_id, agent_id, customer_phone, "
                "call_timestamp, status, created_at "
                "FROM recordings "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            items = [dict(r) for r in rows]
    except Exception as e:
        logger.warning("查询上传历史失败: %s", e)

    return JSONResponse(content={"success": True, "data": items})
