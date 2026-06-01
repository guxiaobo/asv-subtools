"""
Speaker verification endpoint.

POST /api/verify — dual-mode endpoint supporting:
- Mode A (direct): multipart/form-data with two audio files
- Mode B (indirect): JSON body with audio IDs
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from schemas import (
    AudioInputMode,
    EmbeddingInfo,
    EmbeddingSource,
    Scenario,
    ScoringMethod,
    VerifyDirectRequest,
    VerifyIndirectRequest,
    VerifyResponse,
)
from services.verifier import SpeakerVerifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["verification"])


# ---------------------------------------------------------------------------
# Lazy dependency
# ---------------------------------------------------------------------------
_verifier_instance: Optional[SpeakerVerifier] = None


def _get_verifier() -> SpeakerVerifier:
    if _verifier_instance is None:
        raise RuntimeError("Verifier not initialized")
    return _verifier_instance


def init_verifier(v: SpeakerVerifier) -> None:
    global _verifier_instance
    _verifier_instance = v


# ---------------------------------------------------------------------------
# POST /api/verify — direct binary upload (multipart)
# ---------------------------------------------------------------------------
@router.post("/verify", response_model=VerifyResponse)
async def verify(
    # Direct mode: files
    audio_a: Optional[UploadFile] = File(default=None, description="Speaker A audio file"),
    audio_b: Optional[UploadFile] = File(default=None, description="Speaker B audio file"),
    # Indirect mode: IDs
    audio_id_a: Optional[str] = Form(default=None, description="Speaker A audio ID"),
    audio_id_b: Optional[str] = Form(default=None, description="Speaker B audio ID"),
    storage_backend_a: Optional[str] = Form(default="nas", description="Storage for speaker A"),
    storage_backend_b: Optional[str] = Form(default="nas", description="Storage for speaker B"),
    # Shared parameters
    scenario: Optional[str] = Form(default=None, description="Business scenario"),
    threshold: Optional[float] = Form(default=None, ge=0.0, le=1.0, description="Decision threshold"),
    scoring_method: Optional[str] = Form(default=None, description="Scoring method"),
    # Extra (from indirect JSON body)
    bucket_a: Optional[str] = Form(default=None, description="S3 bucket for speaker A"),
    bucket_b: Optional[str] = Form(default=None, description="S3 bucket for speaker B"),
) -> VerifyResponse:
    """
    Verify whether two audio samples belong to the same speaker.

    **Direct mode** (files):
    - Upload `audio_a` and `audio_b` as multipart/form-data files.
    - Supported formats: WAV, ulaw, alaw, raw PCM, FLAC, OGG, MP3.

    **Indirect mode** (IDs):
    - Provide `audio_id_a` and `audio_id_b`.
    - Server fetches audio from NAS / S3 / Redis by ID.
    - Use `storage_backend_a` / `storage_backend_b` to specify backend.

    **Parameters:**
    - `scenario`: `customer_service` | `debt_collection` | `audit` | `custom`
    - `threshold`: Override the decision threshold (overrides scenario default).
    - `scoring_method`: `cosine` | `euclidean` | `dot_product`

    Returns similarity score + binary decision.
    """
    verifier = _get_verifier()

    # Determine mode
    has_files = audio_a is not None and audio_b is not None
    has_ids = audio_id_a is not None and audio_id_b is not None

    if not has_files and not has_ids:
        debug_info = f"audio_a_type={type(audio_a).__name__ if audio_a else None}, audio_b_type={type(audio_b).__name__ if audio_b else None}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provide either audio_a+audio_b (files) or audio_id_a+audio_id_b (IDs). Debug: {debug_info}",
        )

    if has_files:
        # Mode A: Direct binary upload
        assert audio_a is not None and audio_b is not None  # validated above
        audio_bytes_a = await audio_a.read()
        audio_bytes_b = await audio_b.read()

        if len(audio_bytes_a) < 44 or len(audio_bytes_b) < 44:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio files too small (minimum 44 bytes for WAV header)",
            )

        result = verifier.verify_from_bytes(
            audio_bytes_a=audio_bytes_a,
            audio_bytes_b=audio_bytes_b,
            threshold=threshold,
            scoring_method=scoring_method,
            scenario=scenario,
            audio_id_a=audio_a.filename,
            audio_id_b=audio_b.filename,
        )

    else:
        # Mode B: Indirect ID-based retrieval
        assert audio_id_a is not None and audio_id_b is not None  # validated above
        result = verifier.verify_from_ids(
            audio_id_a=audio_id_a,
            audio_id_b=audio_id_b,
            backend_a=storage_backend_a or "nas",
            backend_b=storage_backend_b or "nas",
            threshold=threshold,
            scoring_method=scoring_method,
            scenario=scenario,
            bucket_a=bucket_a,
            bucket_b=bucket_b,
        )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error,
        )

    return result


# ---------------------------------------------------------------------------
# Alias endpoint for JSON-only indirect verification
# ---------------------------------------------------------------------------
@router.post("/verify/indirect", response_model=VerifyResponse)
async def verify_indirect(
    request: VerifyIndirectRequest,
) -> VerifyResponse:
    """
    Verify two speakers by audio ID (JSON body version).

    This endpoint accepts a pure JSON body, making it easier for
    programmatic clients that don't want multipart encoding.

    Example:
        {
            "mode": "indirect",
            "audio_a": {
                "audio_id": "recording-001",
                "storage_backend": "nas"
            },
            "audio_b": {
                "audio_id": "recording-002",
                "storage_backend": "nas"
            },
            "scenario": "debt_collection",
            "threshold": 0.7
        }
    """
    verifier = _get_verifier()

    result = verifier.verify_from_ids(
        audio_id_a=request.audio_a.audio_id,
        audio_id_b=request.audio_b.audio_id,
        backend_a=request.audio_a.storage_backend.value,
        backend_b=request.audio_b.storage_backend.value,
        threshold=request.threshold,
        scoring_method=request.scoring_method.value if request.scoring_method else None,
        scenario=request.scenario.value if request.scenario else None,
        bucket_a=request.audio_a.bucket,
        bucket_b=request.audio_b.bucket,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error,
        )

    return result
