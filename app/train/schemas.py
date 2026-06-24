"""
训练模块的 Pydantic 模型定义。

定义录音状态、预处理结果、训练结果等数据模型，
在 API ⇄ 训练模块之间提供一致的序列化格式。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class BizSystem(str, Enum):
    COLLECTION = "collection"
    CS = "cs"


class AudioSourceType(str, Enum):
    BINARY = "binary"
    URL = "url"
    ID = "id"


class RecordingStatus(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    TRAINED = "trained"


class PreprocessStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class TrainStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# SQLite Recordings 表对应的模型
# ---------------------------------------------------------------------------
class RecordingIn(BaseModel):
    """API 推送录音时的请求数据。"""
    biz_system: BizSystem
    agent_id: str = Field(..., min_length=1, max_length=64)
    customer_id: str = Field(..., min_length=1, max_length=32)
    call_timestamp: str  # ISO 8601
    call_id: str = Field(..., min_length=1, max_length=128)
    audio_source_type: AudioSourceType
    audio_url: Optional[str] = None
    audio_id: Optional[str] = None
    channel_separated: bool = False
    duration_sec: Optional[float] = None


class RecordingOut(BaseModel):
    """录音记录输出（返回给调用方）。"""
    id: int
    call_id: str
    local_audio_path: str
    status: RecordingStatus
    created_at: str


class RecordingRow(BaseModel):
    """录音表的一行完整数据。"""
    id: int
    biz_system: str
    call_id: str
    agent_id: str
    customer_id: str
    call_timestamp: str
    channel_separated: int
    duration_sec: Optional[float]
    audio_source_type: str
    audio_original_url: Optional[str]
    local_audio_path: Optional[str]
    status: str
    pre_status: str
    pre_result: Optional[str]
    pre_error: Optional[str]
    pre_finished_at: Optional[str]
    train_status: str
    train_result: Optional[str]
    train_error: Optional[str]
    train_finished_at: Optional[str]
    model_version: Optional[str]
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# 预处理结果
# ---------------------------------------------------------------------------
class PreprocessResult(BaseModel):
    """每条录音的预处理结果（JSON 结构）。"""
    segment_count: int = 0
    agent_segments: int = 0
    customer_segments: int = 0
    agent_valid_sec: float = 0.0
    customer_valid_sec: float = 0.0
    avg_snr_db: float = 0.0
    min_snr_db: float = 0.0
    max_snr_db: float = 0.0
    dropped_segments: int = 0
    dropped_reason: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# 模型版本相关
# ---------------------------------------------------------------------------
class ModelVersionIn(BaseModel):
    """模型版本记录。"""
    version: str
    eval_metric: str
    eval_value: float
    prev_eval_value: Optional[float]
    improved: bool
    train_recording_count: int
    train_speaker_count: int
    train_time_sec: float
    previous_version: Optional[str]
    model_path: str
    model_md5: Optional[str]
    notes: Optional[str]


class ModelVersionOut(BaseModel):
    """模型版本记录输出。"""
    id: int
    version: str
    eval_metric: str
    eval_value: float
    improved: bool
    is_active: bool
    created_at: str


# ---------------------------------------------------------------------------
# 训练汇总
# ---------------------------------------------------------------------------
class TrainSummary(BaseModel):
    """一轮增量训练的汇总信息。"""
    total_pending: int
    total_agents: int
    total_segments: int
    new_model_version: Optional[str]
    new_eer: Optional[float]
    prev_eer: Optional[float]
    improved: Optional[bool]
    published: bool
    duration_sec: float
