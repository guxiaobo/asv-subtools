"""
Core speaker verification service.

Orchestrates:
1. Audio loading and preprocessing
2. Embedding extraction via ONNX model
3. Embedding cache (check cache first, compute on miss)
4. Similarity scoring
5. Threshold-based decision
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import AppConfig, VerificationConfig
from onnx_model import ONNXModel
from services.audio import AudioData, AudioLoader, InsufficientAudioError
from services.cache import EmbeddingCache, CacheMiss, create_cache
from schemas import (
    AudioInfo,
    EmbeddingInfo,
    EmbeddingSource,
    ScoringMethod,
    Scenario,
    VerifyResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Model input/output name conventions (varies by architecture)
_CAMPPUS_INPUT = "feats"
_CAMPPUS_OUTPUT = "embedding"


class SpeakerVerifier:
    """
    Core speaker verification service.

    Wires up audio loading → ONNX inference → scoring → decision.
    Handles both direct (binary upload) and indirect (ID-based) modes.
    """

    def __init__(
        self,
        config: AppConfig,
        model: ONNXModel,
        audio_loader: AudioLoader,
        cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self._config = config
        self._model = model
        self._audio_loader = audio_loader
        self._ver_config = config.verification
        self._cache = cache

        log_dim = "unknown"
        if model.input_shapes:
            # Get embedding dimension from output shape
            try:
                out_meta = model.get_input_meta(_CAMPPUS_INPUT)
                if out_meta:
                    log_dim = str(out_meta[0])
            except Exception:
                pass

        logger.info(
            "SpeakerVerifier initialized: model=%s provider=%s threshold=%.3f method=%s",
            config.model.path,
            config.model.provider,
            self._ver_config.default_threshold,
            self._ver_config.scoring_method,
        )

    # ------------------------------------------------------------------
    # Audio source resolution (for the router)
    # ------------------------------------------------------------------

    @property
    def audio_loader(self) -> AudioLoader:
        """Expose the underlying audio loader (read-only)."""
        return self._audio_loader

    def resolve_audio_to_bytes(
        self,
        *,
        url: Optional[str] = None,
        audio_id: Optional[str] = None,
        backend: str = "nas",
        bucket: Optional[str] = None,
    ) -> bytes:
        """
        Resolve a synchronous audio source to WAV bytes.

        Priority within this method: ``url`` > ``audio_id``.
        Does **not** handle file uploads (those are async and resolved
        in the router).

        Returns:
            WAV bytes suitable for ``verify_from_bytes``.

        Raises:
            AudioLoadError: If no source resolves.
        """
        from services.audio import AudioLoadError

        if url is not None:
            return self._audio_loader.download_url(url)
        if audio_id is not None:
            audio = self._audio_loader.load_by_id(audio_id, backend=backend, bucket=bucket)
            return audio.to_wav_bytes()
        raise AudioLoadError("No audio source provided (url or audio_id)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_from_bytes(
        self,
        audio_bytes_a: bytes,
        audio_bytes_b: bytes,
        threshold: Optional[float] = None,
        scoring_method: Optional[str] = None,
        scenario: Optional[str] = None,
        audio_id_a: Optional[str] = None,
        audio_id_b: Optional[str] = None,
    ) -> VerifyResponse:
        """
        Verify two speakers from raw audio bytes (direct upload).

        Args:
            audio_bytes_a: Raw audio data for the first speaker.
            audio_bytes_b: Raw audio data for the second speaker.
            threshold: Override the default decision threshold.
            scoring_method: Override the default scoring method.
            scenario: Business scenario for threshold selection.
            audio_id_a: Optional ID for speaker A (for cache lookup).
            audio_id_b: Optional ID for speaker B (for cache lookup).

        Returns:
            VerifyResponse with score and decision.
        """
        start_time = time.perf_counter()

        # 1. Load and preprocess audio
        try:
            audio_a = self._audio_loader.load_from_bytes(audio_bytes_a, audio_id=audio_id_a)
            audio_b = self._audio_loader.load_from_bytes(audio_bytes_b, audio_id=audio_id_b)
            audio_a = self._audio_loader.preprocess(audio_a)
            audio_b = self._audio_loader.preprocess(audio_b)
        except InsufficientAudioError as e:
            return _error_response(f"Insufficient audio: {e}", start_time, "INSUFFICIENT_AUDIO")
        except Exception as e:
            logger.exception("Audio loading failed")
            return _error_response(f"Audio load error: {e}", start_time, "AUDIO_LOAD_ERROR")

        # 2. Extract embeddings (with cache)
        try:
            emb_a, source_a = self._get_embedding(audio_a, cache_key=audio_id_a)
            emb_b, source_b = self._get_embedding(audio_b, cache_key=audio_id_b)
        except Exception as e:
            logger.exception("Embedding extraction failed")
            return _error_response(f"Embedding error: {e}", start_time, "EMBEDDING_ERROR")

        # 3. Score
        method = ScoringMethod(scoring_method) if scoring_method else \
                 ScoringMethod(self._ver_config.scoring_method)
        score = self._compute_score(emb_a, emb_b, method)
        # Floating-point guard: clamp cosine similarity to [-1, 1]
        score = max(-1.0, min(1.0, score))

        # 4. Threshold decision
        threshold = _resolve_threshold(threshold, scenario, self._config)
        is_same = score >= threshold if threshold is not None else False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return VerifyResponse(
            success=True,
            is_same_speaker=is_same,
            score=float(score),
            threshold_used=threshold or 0.0,
            processing_time_ms=round(elapsed_ms, 2),
            embedding_a=EmbeddingInfo(
                dimension=len(emb_a),
                source=source_a,
                norm=float(np.linalg.norm(emb_a)),
            ),
            embedding_b=EmbeddingInfo(
                dimension=len(emb_b),
                source=source_b,
                norm=float(np.linalg.norm(emb_b)),
            ),
            audio_a=AudioInfo(
                duration_sec=audio_a.duration_sec,
                sample_rate=audio_a.sample_rate,
                valid_speech_sec=audio_a.duration_sec,
                channels=1,
            ),
            audio_b=AudioInfo(
                duration_sec=audio_b.duration_sec,
                sample_rate=audio_b.sample_rate,
                valid_speech_sec=audio_b.duration_sec,
                channels=1,
            ),
            scenario=scenario,
        )

    def verify_from_ids(
        self,
        audio_id_a: str,
        audio_id_b: str,
        backend_a: str = "nas",
        backend_b: str = "nas",
        threshold: Optional[float] = None,
        scoring_method: Optional[str] = None,
        scenario: Optional[str] = None,
        bucket_a: Optional[str] = None,
        bucket_b: Optional[str] = None,
    ) -> VerifyResponse:
        """
        Verify two speakers by audio ID (indirect retrieval).

        Args:
            audio_id_a: ID for speaker A's audio.
            audio_id_b: ID for speaker B's audio.
            backend_a: Storage backend for speaker A.
            backend_b: Storage backend for speaker B.
            threshold: Override threshold.
            scoring_method: Override scoring method.
            scenario: Business scenario.
            bucket_a: S3 bucket for speaker A.
            bucket_b: S3 bucket for speaker B.

        Returns:
            VerifyResponse.
        """
        start_time = time.perf_counter()

        try:
            audio_a = self._audio_loader.load_by_id(audio_id_a, backend=backend_a, bucket=bucket_a)
            audio_b = self._audio_loader.load_by_id(audio_id_b, backend=backend_b, bucket=bucket_b)
        except Exception as e:
            logger.exception("Audio ID fetch failed")
            return _error_response(f"Audio fetch error: {e}", start_time, "AUDIO_FETCH_ERROR")

        # Compute embeddings directly from AudioData (avoids re-encode roundtrip)
        try:
            emb_a, source_a = self._get_embedding(
                audio_a, cache_key=audio_id_a
            )
            emb_b, source_b = self._get_embedding(
                audio_b, cache_key=audio_id_b
            )
        except Exception as e:
            logger.exception("Embedding extraction failed")
            return _error_response(
                f"Embedding error: {e}", start_time, "EMBEDDING_ERROR"
            )

        # Score + threshold
        method = ScoringMethod(scoring_method) if scoring_method else \
                 ScoringMethod(self._ver_config.scoring_method)
        score = self._compute_score(emb_a, emb_b, method)
        score = max(-1.0, min(1.0, score))

        threshold = _resolve_threshold(threshold, scenario, self._config)
        is_same = score >= threshold if threshold is not None else False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return VerifyResponse(
            success=True,
            is_same_speaker=is_same,
            score=float(score),
            threshold_used=float(threshold) if threshold is not None else 0.0,
            processing_time_ms=elapsed_ms,
            embedding_a=EmbeddingInfo(
                source=source_a,
                dimension=len(emb_a),
            ),
            embedding_b=EmbeddingInfo(
                source=source_b,
                dimension=len(emb_b),
            ),
            audio_a=AudioInfo(
                duration_sec=audio_a.duration_sec,
                sample_rate=audio_a.sample_rate,
                valid_speech_sec=audio_a.duration_sec,
                channels=1,
            ),
            audio_b=AudioInfo(
                duration_sec=audio_b.duration_sec,
                sample_rate=audio_b.sample_rate,
                valid_speech_sec=audio_b.duration_sec,
                channels=1,
            ),
            scenario=scenario,
            elapsed_ms=elapsed_ms,
        )

    def verify_from_urls(
        self,
        url_a: str,
        url_b: str,
        threshold: Optional[float] = None,
        scoring_method: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> VerifyResponse:
        """
        Verify two speakers by downloading audio from HTTP(S) URLs.

        Delegates to ``download_url`` on the audio loader (which caches
        downloads locally keyed by MD5 of the URL), then processes the
        bytes through the standard path.

        Args:
            url_a: HTTP/HTTPS URL for speaker A's audio.
            url_b: HTTP/HTTPS URL for speaker B's audio.
            threshold: Override the default decision threshold.
            scoring_method: Override the default scoring method.
            scenario: Business scenario for threshold selection.

        Returns:
            VerifyResponse with score and decision.
        """
        start_time = time.perf_counter()

        # Download
        try:
            audio_bytes_a = self._audio_loader.download_url(url_a)
            audio_bytes_b = self._audio_loader.download_url(url_b)
        except Exception as e:
            logger.exception("URL download failed")
            return _error_response(
                f"Audio download error: {e}", start_time, "AUDIO_DOWNLOAD_ERROR"
            )

        url_hash_a = hashlib.md5(url_a.encode("utf-8")).hexdigest()
        url_hash_b = hashlib.md5(url_b.encode("utf-8")).hexdigest()

        # Delegate to the bytes path (load → preprocess → embed → score)
        return self.verify_from_bytes(
            audio_bytes_a=audio_bytes_a,
            audio_bytes_b=audio_bytes_b,
            threshold=threshold,
            scoring_method=scoring_method,
            scenario=scenario,
            audio_id_a=url_hash_a,
            audio_id_b=url_hash_b,
        )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _get_embedding(
        self,
        audio: AudioData,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, EmbeddingSource]:
        """
        Extract speaker embedding: check cache first, compute on miss.

        Returns:
            (embedding_vector, source: computed|cached)
        """
        source = EmbeddingSource.COMPUTED

        # Check cache
        if cache_key is not None and self._cache and self._cache.enabled:
            try:
                emb = self._cache.get(cache_key)
                logger.debug("Cache HIT for '%s'", cache_key)
                return emb, EmbeddingSource.CACHED
            except CacheMiss:
                logger.debug("Cache MISS for '%s', computing", cache_key)

        # Compute
        emb = self._compute_embedding(audio)

        # Store in cache
        if cache_key is not None and self._cache and self._cache.enabled:
            self._cache.set(cache_key, emb)

        return emb, source

    def _compute_embedding(self, audio: AudioData) -> np.ndarray:
        """
        Run ONNX model to produce a speaker embedding vector.

        1. Extract fbank features
        2. Prepare input tensor
        3. Run ONNX session
        4. Return embedding (L2-normalized if configured)
        """
        # Extract fbank features
        feat_config = self._config.audio
        fbank = self._audio_loader.extract_fbank(
            audio,
            num_filters=feat_config.fbank_num_filters,
            window_ms=feat_config.fbank_window_ms,
            hop_ms=feat_config.fbank_hop_ms,
        )  # shape: (T, num_filters)

        # Add batch and channel dims -> (1, T, num_filters) or (1, 1, T, num_filters)
        # CAM++ expects (1, T, F) where T=time, F=filterbank
        if fbank.ndim == 2:
            input_tensor = fbank[np.newaxis, ...]  # (1, T, F)
        else:
            input_tensor = fbank

        # Determine input name from ONNX model metadata
        # CAM++ uses "feats"; other models may use "input", "input.1", etc.
        primary_input = _CAMPPUS_INPUT
        if primary_input not in self._model.input_names:
            # Auto-detect: use the first available input name
            primary_input = self._model.input_names[0]
            logger.debug("Auto-detected model primary input name: %s", primary_input)

        # Build input dict: primary features plus any auxiliary inputs
        feed_dict: Dict[str, np.ndarray] = {}
        for inp_name in self._model.input_names:
            if inp_name == primary_input:
                feed_dict[inp_name] = input_tensor.astype(np.float32)
            elif inp_name.lower() in ("feature_lens", "input_lengths", "lengths"):
                # Audio length in frames: (batch,)
                # Use expected dtype from model metadata
                num_frames = input_tensor.shape[1]
                inp_meta = self._model.get_input_meta(inp_name)
                dt = _onnx_dtype_to_numpy(inp_meta[1]) if inp_meta else np.float32
                feed_dict[inp_name] = np.array([num_frames], dtype=dt)
            elif inp_name.lower() in ("state", "state_c", "state_h"):
                # RNN/LSTM states — skip (model should handle default)
                logger.debug("Skipping optional input %s (model should use default)", inp_name)
                continue
            else:
                # Unknown auxiliary input — pass zero of expected shape
                inp_meta = self._model.get_input_meta(inp_name)
                if inp_meta:
                    shape = inp_meta[0]
                    if shape is not None:
                        try:
                            concrete = [1 if isinstance(d, str) else (d or 1) for d in shape]
                            feed_dict[inp_name] = np.zeros(concrete, dtype=np.float32)
                        except Exception:
                            pass

        # Run inference
        try:
            outputs = self._model.infer(feed_dict)
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}") from e

        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs")

        embedding = outputs[0]  # shape: (1, D) or (D,)
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        # L2 normalize if configured
        if self._ver_config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def _compute_score(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        method: ScoringMethod,
    ) -> float:
        """Compute similarity score between two embeddings."""
        if method == ScoringMethod.COSINE:
            return self._cosine_similarity(emb_a, emb_b)
        elif method == ScoringMethod.EUCLIDEAN:
            return self._euclidean_similarity(emb_a, emb_b)
        elif method == ScoringMethod.DOT_PRODUCT:
            return float(np.dot(emb_a, emb_b))
        else:
            logger.warning("Unknown scoring method %s, using cosine", method)
            return self._cosine_similarity(emb_a, emb_b)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors. Range: [-1, 1]."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean similarity: 1 / (1 + distance). Range: (0, 1]."""
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def _onnx_dtype_to_numpy(onnx_type_str: str) -> np.dtype:
    """Convert ONNX type string (e.g. 'tensor(float)') to numpy dtype."""
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(onnx_type_str.lower(), np.float32)


def _compute_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    method: str,
) -> float:
    """Compute similarity score between two embeddings (standalone)."""
    if method == "cosine":
        emb_a_n = emb_a / (np.linalg.norm(emb_a) + 1e-10)
        emb_b_n = emb_b / (np.linalg.norm(emb_b) + 1e-10)
        return float(np.dot(emb_a_n, emb_b_n))
    elif method == "euclidean":
        dist = float(np.linalg.norm(emb_a - emb_b))
        return float(1.0 / (1.0 + dist))
    elif method == "dot_product":
        return float(np.dot(emb_a, emb_b))
    else:
        return float(np.dot(emb_a, emb_b))


# ------------------------------------------------------------------
# Threshold resolution
# ------------------------------------------------------------------


def _resolve_threshold(
    override: Optional[float],
    scenario: Optional[str],
    config: 'AppConfig',
) -> Optional[float]:
    """
    Resolve the decision threshold.

    Priority:
    1. Explicit override (from request)
    2. Scenario-specific threshold (from config)
    3. Default threshold (from config)
    """
    from config import AppConfig
    if override is not None:
        return override
    if scenario is not None:
        scenario_threshold = config.get_scenario_threshold(scenario)
        if scenario_threshold is not None:
            return scenario_threshold
    return config.verification.default_threshold


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _audio_bytes_from_data(audio: "AudioData") -> bytes:
    """Convert AudioData back to bytes (for the shared code path)."""
    import struct
    pcm = (audio.waveform * 32767.0).astype(np.int16).tobytes()
    return pcm


def _error_response(
    message: str,
    start_time: float,
    code: str = "ERROR",
) -> VerifyResponse:
    """Build an error VerifyResponse."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return VerifyResponse(
        success=False,
        is_same_speaker=False,
        score=0.0,
        threshold_used=0.0,
        processing_time_ms=round(elapsed_ms, 2),
        embedding_a=EmbeddingInfo(dimension=0, source=EmbeddingSource.COMPUTED),
        embedding_b=EmbeddingInfo(dimension=0, source=EmbeddingSource.COMPUTED),
        error=f"[{code}] {message}",
    )
