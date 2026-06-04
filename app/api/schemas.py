"""
Pydantic schemas for the ASV verification API.

Defines request/response models, validation, and error types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AudioInputMode(str, Enum):
    """How audio is provided to the API."""

    DIRECT = "direct"
    INDIRECT = "indirect"


class StorageBackend(str, Enum):
    """Backend storage for indirect audio retrieval."""

    NAS = "nas"
    S3 = "s3"
    REDIS = "redis"


class ScoringMethod(str, Enum):
    """Similarity scoring method."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class Scenario(str, Enum):
    """Business scenario that determines threshold selection."""

    CUSTOMER_SERVICE = "customer_service"
    DEBT_COLLECTION = "debt_collection"
    AUDIT = "audit"
    CUSTOM = "custom"


class EmbeddingSource(str, Enum):
    """Where the embedding came from."""

    COMPUTED = "computed"
    CACHED = "cached"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class VerifyDirectRequest(BaseModel):
    """Direct audio upload — files come via multipart form, not JSON."""

    mode: AudioInputMode = AudioInputMode.DIRECT
    scenario: Optional[Scenario] = None
    threshold: Optional[float] = None
    scoring_method: Optional[ScoringMethod] = None


class IndirectAudioRef(BaseModel):
    """Reference to an audio file stored externally or accessible via URL."""

    audio_id: Optional[str] = Field(
        default=None, min_length=1, max_length=256,
        description="Unique audio identifier (omit if url is provided)",
    )
    storage_backend: StorageBackend = Field(default=StorageBackend.NAS)
    bucket: Optional[str] = Field(default=None, description="S3 bucket name (required for s3 backend)")
    format: Optional[str] = Field(default=None, description="Audio format hint, e.g. wav, ulaw")
    url: Optional[str] = Field(default=None, description="HTTP/HTTPS URL to download audio from")

    @model_validator(mode="after")
    def check_audio_id_or_url(self) -> "IndirectAudioRef":
        if not self.audio_id and not self.url:
            raise ValueError("Either audio_id or url must be provided")
        if self.audio_id and self.url:
            raise ValueError("Provide either audio_id or url, not both")
        return self

    @model_validator(mode="after")
    def check_audio_ids_differ(self) -> "IndirectAudioRef":
        # This method exists only for the parent model validator;
        # it's a no-op on individual refs.
        return self


class VerifyIndirectRequest(BaseModel):
    """Request body for indirect (ID-based) audio verification."""

    mode: AudioInputMode = AudioInputMode.INDIRECT
    audio_a: IndirectAudioRef
    audio_b: IndirectAudioRef
    scenario: Optional[Scenario] = None
    threshold: Optional[float] = None
    scoring_method: Optional[ScoringMethod] = None
    cache_ttl_sec: Optional[int] = Field(default=None, ge=60, le=2592000, description="Override cache TTL")

    @model_validator(mode="after")
    def check_audio_refs_differ(self) -> "VerifyIndirectRequest":
        # Compare by the same reference type: both URLs or both IDs
        if self.audio_a.url and self.audio_b.url:
            if self.audio_a.url == self.audio_b.url:
                raise ValueError("audio_a and audio_b must have different URLs")
        elif self.audio_a.audio_id and self.audio_b.audio_id:
            if self.audio_a.audio_id == self.audio_b.audio_id:
                raise ValueError("audio_a and audio_b must have different IDs")
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class EmbeddingInfo(BaseModel):
    """Details about the computed embedding."""

    dimension: int
    source: EmbeddingSource
    norm: Optional[float] = None


class AudioInfo(BaseModel):
    """Details about processed audio."""

    duration_sec: float
    sample_rate: int
    valid_speech_sec: Optional[float] = None
    channels: int = 1


class VerifyResponse(BaseModel):
    """Response from the verify endpoint."""

    success: bool = True
    is_same_speaker: bool
    score: float = Field(..., ge=-1.0, le=1.0 + 1e-6)
    threshold_used: float
    processing_time_ms: float
    embedding_a: EmbeddingInfo
    embedding_b: EmbeddingInfo
    audio_a: Optional[AudioInfo] = None
    audio_b: Optional[AudioInfo] = None
    scenario: Optional[str] = None
    error: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    model_loaded: bool
    model_path: str
    model_provider: str
    uptime_sec: float
    cache_connected: bool


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Metrics snapshot (internal, for Prometheus export)
# ---------------------------------------------------------------------------
class MetricsSnapshot(BaseModel):
    """Snapshot of service metrics for Prometheus endpoint."""

    total_requests: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    insufficient_audio: int = 0
    average_processing_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    active_model_path: str = ""
    by_scenario: Dict[str, int] = {}
