"""
Speaker verification endpoint.

POST /api/verify — dual-mode endpoint supporting:
- Mode A (direct): multipart/form-data with two audio files
- Mode B (indirect): JSON body with audio IDs
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from schemas import (
    AudioInputMode,
    EmbeddingInfo,
    EmbeddingSource,
    IndirectAudioRef,
    Scenario,
    ScoringMethod,
    VerifyDirectRequest,
    VerifyIndirectRequest,
    VerifyResponse,
)
from services.recording_db import log_api_call
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
# Helper: resolve a single audio source to bytes
# ---------------------------------------------------------------------------
async def _resolve_single_audio(
    verifier: SpeakerVerifier,
    audio_file: Optional[UploadFile],
    url: Optional[str],
    audio_id: Optional[str],
    backend: str,
    bucket: Optional[str],
    label: str,
) -> Tuple[bytes, Optional[str]]:
    """
    Resolve one audio to WAV bytes.

    Priority: **file (binary stream) > URL > audio_id (fetcher)**.

    Returns:
        Tuple of (wav_bytes, label).  The label is used as the cache key for
        ``verify_from_bytes`` — it will be ``filename`` for file uploads,
        ``MD5(url)`` for URL downloads, or ``audio_id`` for ID fetches.
    """
    if audio_file is not None:
        data = await audio_file.read()
        if len(data) < 44:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio {label} too small (minimum 44 bytes)",
            )
        return data, audio_file.filename

    if url is not None:
        data = verifier.resolve_audio_to_bytes(url=url)
        cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
        return data, cache_key

    if audio_id is not None:
        data = verifier.resolve_audio_to_bytes(
            audio_id=audio_id, backend=backend, bucket=bucket
        )
        return data, audio_id

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"No source provided for {label}: provide file, url, or audio_id",
    )


# ---------------------------------------------------------------------------
# POST /api/verify — per-audio independent source resolution
# ---------------------------------------------------------------------------
@router.post("/verify", response_model=VerifyResponse)
async def verify(
    request: Request,
    # Direct mode: files
    audio_a: Optional[UploadFile] = File(default=None, description="Speaker A audio file"),
    audio_b: Optional[UploadFile] = File(default=None, description="Speaker B audio file"),
    # URL mode
    url_a: Optional[str] = Form(default=None, description="Speaker A audio URL"),
    url_b: Optional[str] = Form(default=None, description="Speaker B audio URL"),
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

    Each audio is independently resolved with the following priority:

        1. **Binary stream** (file upload) — fastest, lowest latency.
        2. **URL** — server downloads via HTTP(S), caches locally by MD5.
        3. **audio_id** — fetches from NAS / S3 / Redis.

    This means you can mix sources across the two speakers
    (e.g. audio_a as a file, audio_b as a URL).

    **Supported audio formats:** WAV, ulaw, alaw, raw PCM, FLAC, OGG, MP3.

    **Parameters:**
    - ``scenario``: ``customer_service`` | ``debt_collection`` | ``audit`` | ``custom``
    - ``threshold``: Override the decision threshold (overrides scenario default).
    - ``scoring_method``: ``cosine`` | ``euclidean`` | ``dot_product``

    Returns similarity score + binary decision.
    """
    verifier = _get_verifier()
    t0 = time.time()

    # Resolve each audio independently (file > url > id)
    bytes_a, label_a = await _resolve_single_audio(
        verifier, audio_a, url_a, audio_id_a,
        storage_backend_a or "nas", bucket_a, "A",
    )
    bytes_b, label_b = await _resolve_single_audio(
        verifier, audio_b, url_b, audio_id_b,
        storage_backend_b or "nas", bucket_b, "B",
    )

    result = verifier.verify_from_bytes(
        audio_bytes_a=bytes_a,
        audio_bytes_b=bytes_b,
        threshold=threshold,
        scoring_method=scoring_method,
        scenario=scenario,
        audio_id_a=label_a,
        audio_id_b=label_b,
    )

    duration_ms = int((time.time() - t0) * 1000) if "t0" in dir() else 0

    if not result.success:
        # Log the failure
        _determine_source = lambda f, u, i: ("file", f.filename if f else "") if f else (("url", u or "") if u else ("audio_id", i or ""))
        a_src, a_val = _determine_source(audio_a, url_a, audio_id_a)
        b_src, b_val = _determine_source(audio_b, url_b, audio_id_b)
        await log_api_call(
            endpoint="/api/verify",
            audio_a_source=a_src, audio_a_value=a_val,
            audio_b_source=b_src, audio_b_value=b_val,
            has_audio_data=audio_a is not None or audio_b is not None,
            duration_ms=duration_ms,
            scenario=scenario,
            caller_ip=request.client.host if request.client else None,
            error_detail=result.error,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error,
        )

    # Log the successful call
    a_src, a_val = (("file", audio_a.filename if audio_a else "") if audio_a else (("url", url_a or "") if url_a else ("audio_id", audio_id_a or "")))
    b_src, b_val = (("file", audio_b.filename if audio_b else "") if audio_b else (("url", url_b or "") if url_b else ("audio_id", audio_id_b or "")))
    await log_api_call(
        endpoint="/api/verify",
        audio_a_source=a_src, audio_a_value=a_val,
        audio_b_source=b_src, audio_b_value=b_val,
        has_audio_data=audio_a is not None or audio_b is not None,
        duration_ms=duration_ms,
        score=result.score,
        decision="same" if result.match else "different",
        threshold=result.threshold,
        scenario=scenario,
        caller_ip=request.client.host if request.client else None,
    )

    return result


# ---------------------------------------------------------------------------
# Helper: resolve a single IndirectAudioRef to bytes
# ---------------------------------------------------------------------------
def _resolve_single_audio_indirect(
    verifier: SpeakerVerifier,
    ref: IndirectAudioRef,
    label: str,
) -> Tuple[bytes, Optional[str]]:
    """
    Resolve one ``IndirectAudioRef`` to WAV bytes.

    Priority: **URL > audio_id** (files are not possible in JSON mode).
    """
    if ref.url is not None:
        data = verifier.resolve_audio_to_bytes(url=ref.url)
        cache_key = hashlib.md5(ref.url.encode("utf-8")).hexdigest()
        return data, cache_key

    if ref.audio_id is not None:
        data = verifier.resolve_audio_to_bytes(
            audio_id=ref.audio_id,
            backend=ref.storage_backend.value if ref.storage_backend else "nas",
            bucket=ref.bucket,
        )
        return data, ref.audio_id

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"No source provided for {label} (provide url or audio_id)",
    )


# ---------------------------------------------------------------------------
# Alias endpoint for JSON-only indirect verification
# ---------------------------------------------------------------------------
@router.post("/verify/indirect", response_model=VerifyResponse)
async def verify_indirect(
    request: Request,
    request_body: VerifyIndirectRequest,
) -> VerifyResponse:
    """
    Verify two speakers by audio ID or URL (JSON body version).

    This endpoint accepts a pure JSON body, making it easier for
    programmatic clients that don't want multipart encoding.

    Example (by ID):
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

    Example (by URL):
        {
            "audio_a": {
                "url": "https://example.com/speaker_a.wav"
            },
            "audio_b": {
                "url": "https://example.com/speaker_b.wav"
            },
            "scenario": "customer_service"
        }
    """
    verifier = _get_verifier()
    t0 = time.time()

    # Per-audio resolution for JSON body as well
    bytes_a, label_a = _resolve_single_audio_indirect(
        verifier, request_body.audio_a, "A",
    )
    bytes_b, label_b = _resolve_single_audio_indirect(
        verifier, request_body.audio_b, "B",
    )

    result = verifier.verify_from_bytes(
        audio_bytes_a=bytes_a,
        audio_bytes_b=bytes_b,
        threshold=request_body.threshold,
        scoring_method=request_body.scoring_method.value if request_body.scoring_method else None,
        scenario=request_body.scenario.value if request_body.scenario else None,
        audio_id_a=label_a,
        audio_id_b=label_b,
    )

    duration_ms = int((time.time() - t0) * 1000)

    def _ref_source(ref):
        return ("url", ref.url) if ref.url else ("audio_id", ref.audio_id or "")

    a_src, a_val = _ref_source(request_body.audio_a)
    b_src, b_val = _ref_source(request_body.audio_b)

    if not result.success:
        await log_api_call(
            endpoint="/api/verify/indirect",
            audio_a_source=a_src, audio_a_value=a_val,
            audio_b_source=b_src, audio_b_value=b_val,
            duration_ms=duration_ms,
            scenario=request_body.scenario.value if request_body.scenario else None,
            caller_ip=request.client.host if request.client else None,
            error_detail=result.error,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error,
        )

    await log_api_call(
        endpoint="/api/verify/indirect",
        audio_a_source=a_src, audio_a_value=a_val,
        audio_b_source=b_src, audio_b_value=b_val,
        duration_ms=duration_ms,
        score=result.score,
        decision="same" if result.match else "different",
        threshold=result.threshold,
        scenario=request_body.scenario.value if request_body.scenario else None,
        caller_ip=request.client.host if request.client else None,
    )

    return result
