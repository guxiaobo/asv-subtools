"""
ASV SDK for Python — Speaker Verification API client.

Usage::

    from asv_sdk import ASVClient

    client = ASVClient(base_url="http://localhost:8000", timeout=30)

    # Mode A: direct file upload
    result = client.verify_files(
        "/path/to/speaker_a.wav",
        "/path/to/speaker_b.wav",
        scenario="debt_collection",
    )
    print(result.is_same_speaker, result.score)  # noqa

    # Mode B: indirect by audio ID
    result = client.verify_ids(
        audio_id_a="recording-001",
        audio_id_b="recording-002",
        backend_a="nas",
        scenario="customer_service",
    )
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingInfo:
    dimension: int
    source: str  # "computed" | "cached"
    norm: Optional[float] = None


@dataclass
class AudioInfo:
    duration_sec: float
    sample_rate: int
    valid_speech_sec: Optional[float] = None
    channels: int = 1


@dataclass
class VerifyResult:
    """Parsed response from the /api/verify endpoint."""

    success: bool
    is_same_speaker: bool
    score: float
    threshold_used: float
    processing_time_ms: float
    embedding_a: EmbeddingInfo
    embedding_b: EmbeddingInfo
    audio_a: Optional[AudioInfo] = None
    audio_b: Optional[AudioInfo] = None
    scenario: Optional[str] = None
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifyResult":
        emb_a = EmbeddingInfo(**data.get("embedding_a", {}))
        emb_b = EmbeddingInfo(**data.get("embedding_b", {}))
        audio_a = AudioInfo(**data["audio_a"]) if data.get("audio_a") else None
        audio_b = AudioInfo(**data["audio_b"]) if data.get("audio_b") else None
        return cls(
            success=data.get("success", True),
            is_same_speaker=data.get("is_same_speaker", False),
            score=data.get("score", 0.0),
            threshold_used=data.get("threshold_used", 0.0),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            embedding_a=emb_a,
            embedding_b=emb_b,
            audio_a=audio_a,
            audio_b=audio_b,
            scenario=data.get("scenario"),
            error=data.get("error"),
            raw_response=data,
        )


@dataclass
class HealthResult:
    status: str
    model_loaded: bool
    model_path: str
    model_provider: str
    uptime_sec: float
    cache_connected: bool
    version: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthResult":
        return cls(**data)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ASVError(Exception):
    """Base exception for ASV SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ServerError(ASVError):
    """Server returned an error response."""
    pass


class NetworkError(ASVError):
    """Network connectivity issue."""
    pass


class ValidationError(ASVError):
    """Invalid input parameters."""
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class ASVClient:
    """
    Python client for the ASV Speaker Verification API.

    Supports both direct file upload and indirect ID-based verification,
    with automatic retry, timeout, and comprehensive error handling.

    Args:
        base_url: API base URL (e.g. ``http://localhost:8000``).
        api_key: Optional API key (sent as ``Authorization: Bearer <key>``).
        timeout: Request timeout in seconds.
        max_retries: Number of automatic retries on network errors.
        verify_ssl: Verify SSL certificates.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {"User-Agent": "asv-sdk-python/0.1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
        )

    # ------------------------------------------------------------------
    # Verify — Mode A: Direct file upload
    # ------------------------------------------------------------------

    def verify_files(
        self,
        audio_a: Union[str, Path, bytes],
        audio_b: Union[str, Path, bytes],
        scenario: Optional[str] = None,
        threshold: Optional[float] = None,
        scoring_method: Optional[str] = None,
    ) -> VerifyResult:
        """
        Verify two speakers by uploading audio files (Mode A).

        Args:
            audio_a: Path or bytes for speaker A's audio.
            audio_b: Path or bytes for speaker B's audio.
            scenario: Business scenario (customer_service, debt_collection, audit).
            threshold: Decision threshold override (0.0–1.0).
            scoring_method: Scoring method (cosine, euclidean, dot_product).

        Returns:
            VerifyResult with score and decision.

        Raises:
            FileNotFoundError: If a file path does not exist.
            ValidationError: If input parameters are invalid.
            ServerError: If the server returns an error.
            NetworkError: If a network error occurs.
        """
        files = {
            "audio_a": self._prepare_file("audio_a", audio_a),
            "audio_b": self._prepare_file("audio_b", audio_b),
        }

        data: Dict[str, str] = {}
        if scenario:
            data["scenario"] = scenario
        if threshold is not None:
            data["threshold"] = str(threshold)
        if scoring_method:
            data["scoring_method"] = scoring_method

        response_data = self._request("POST", "/api/verify", files=files, data=data)
        return VerifyResult.from_dict(response_data)

    # ------------------------------------------------------------------
    # Verify — Mode B: Indirect by audio ID
    # ------------------------------------------------------------------

    def verify_ids(
        self,
        audio_id_a: str,
        audio_id_b: str,
        backend_a: str = "nas",
        backend_b: str = "nas",
        scenario: Optional[str] = None,
        threshold: Optional[float] = None,
        scoring_method: Optional[str] = None,
        bucket_a: Optional[str] = None,
        bucket_b: Optional[str] = None,
    ) -> VerifyResult:
        """
        Verify two speakers by audio ID (Mode B — indirect retrieval).

        Args:
            audio_id_a: Audio ID for speaker A.
            audio_id_b: Audio ID for speaker B.
            backend_a: Storage backend for speaker A (nas, s3, redis).
            backend_b: Storage backend for speaker B.
            scenario: Business scenario.
            threshold: Decision threshold override.
            scoring_method: Scoring method.
            bucket_a: S3 bucket for speaker A.
            bucket_b: S3 bucket for speaker B.

        Returns:
            VerifyResult.

        Raises:
            ValidationError: If IDs are invalid.
            ServerError: If server returns an error.
            NetworkError: If a network error occurs.
        """
        payload: Dict[str, Any] = {
            "mode": "indirect",
            "audio_a": {
                "audio_id": audio_id_a,
                "storage_backend": backend_a,
            },
            "audio_b": {
                "audio_id": audio_id_b,
                "storage_backend": backend_b,
            },
        }
        if scenario:
            payload["scenario"] = scenario
        if threshold is not None:
            payload["threshold"] = threshold
        if scoring_method:
            payload["scoring_method"] = scoring_method
        if bucket_a:
            payload["audio_a"]["bucket"] = bucket_a
        if bucket_b:
            payload["audio_b"]["bucket"] = bucket_b

        response_data = self._request("POST", "/api/verify/indirect", json=payload)
        return VerifyResult.from_dict(response_data)

    # ------------------------------------------------------------------
    # Batch verify (multiple comparisons)
    # ------------------------------------------------------------------

    def verify_batch(
        self,
        comparisons: List[Dict[str, Any]],
    ) -> List[VerifyResult]:
        """
        Perform multiple verifications sequentially.

        Each comparison dict should have the same structure as
        ``verify_files`` or ``verify_ids`` parameters, plus a
        ``mode`` key set to ``"files"`` or ``"ids"``.

        Args:
            comparisons: List of comparison configs.

        Returns:
            List of VerifyResult objects in the same order.
        """
        results: List[VerifyResult] = []
        for comp in comparisons:
            mode = comp.get("mode", "files")
            if mode == "files":
                result = self.verify_files(
                    audio_a=comp["audio_a"],
                    audio_b=comp["audio_b"],
                    scenario=comp.get("scenario"),
                    threshold=comp.get("threshold"),
                    scoring_method=comp.get("scoring_method"),
                )
            else:
                result = self.verify_ids(
                    audio_id_a=comp["audio_id_a"],
                    audio_id_b=comp["audio_id_b"],
                    backend_a=comp.get("backend_a", "nas"),
                    backend_b=comp.get("backend_b", "nas"),
                    scenario=comp.get("scenario"),
                    threshold=comp.get("threshold"),
                    scoring_method=comp.get("scoring_method"),
                )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health(self) -> HealthResult:
        """Query the API health endpoint."""
        data = self._request("GET", "/health")
        return HealthResult.from_dict(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_file(
        self,
        field_name: str,
        audio: Union[str, Path, bytes],
    ) -> tuple:
        """Prepare a file tuple for multipart upload."""
        if isinstance(audio, bytes):
            return (f"{field_name}.wav", audio, "audio/wav")
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return (
            path.name,
            path.read_bytes(),
            self._guess_mime(path.suffix),
        )

    @staticmethod
    def _guess_mime(ext: str) -> str:
        mapping = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".ulaw": "audio/basic",
            ".alaw": "audio/basic",
            ".raw": "audio/x-pcm",
        }
        return mapping.get(ext.lower(), "application/octet-stream")

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with automatic retry.

        Raises:
            NetworkError: After exhausting retries.
            ServerError: If the server returns a non-2xx response.
        """
        url = f"{self.base_url}{path}" if not kwargs.get("files") else path
        # For multipart, we need full URL since httpx doesn't resolve base_url
        # properly when files are present
        if kwargs.get("files") and not url.startswith("http"):
            url = f"{self.base_url}{path}"

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    response = self._client.get(url, **kwargs)
                elif method == "POST":
                    response = self._client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Success
                if response.is_success:
                    return response.json()

                # Server returned an error
                try:
                    body = response.json()
                    error_msg = body.get("error") or body.get("detail") or str(body)
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                except Exception:
                    error_msg = response.text or f"HTTP {response.status_code}"

                raise ServerError(
                    f"Server error ({response.status_code}): {error_msg}",
                    status_code=response.status_code,
                )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = NetworkError(f"Network error: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise last_error from e

            except ServerError:
                raise

            except Exception as e:
                if isinstance(e, ASVError):
                    raise
                raise NetworkError(f"Unexpected error: {e}") from e

        # Should not reach here
        raise NetworkError(f"Request failed after {self.max_retries + 1} attempts")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "ASVClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def verify(
    base_url: str = "http://localhost:8000",
    **kwargs: Any,
) -> VerifyResult:
    """
    Quick one-shot verification without creating a client.

    Accepts the same arguments as ``ASVClient.verify_files`` or
    ``ASVClient.verify_ids``, auto-detected by whether ``audio_id_a``
    is provided.

    Example::

        result = verify(audio_a="a.wav", audio_b="b.wav", scenario="audit")
        print(result.is_same_speaker, result.score)  # noqa
    """
    with ASVClient(base_url=base_url) as client:
        if "audio_id_a" in kwargs:
            return client.verify_ids(**kwargs)
        return client.verify_files(**kwargs)
