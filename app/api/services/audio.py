"""
Audio loading and preprocessing service.

Supports:
- Direct binary upload decoding
- ID-based retrieval from NAS / S3 / Redis
- Resampling, VAD, fbank feature extraction
- Audio validation and length checks
"""

from __future__ import annotations

import hashlib
import io
import logging
import struct
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from config import AudioConfig, StorageConfig
from services.fetcher import AudioFetcher, FetchError

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    WAV = "wav"
    ULAW = "ulaw"
    ALAW = "alaw"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


# ---------------------------------------------------------------------------
# Audio data container
# ---------------------------------------------------------------------------
class AudioData:
    """Holds raw audio waveform and metadata."""

    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        audio_id: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ) -> None:
        self.waveform = waveform  # shape: (num_samples,) float32 in [-1, 1]
        self.sample_rate = sample_rate
        self.audio_id = audio_id
        self._duration_sec = duration_sec

    @property
    def duration_sec(self) -> float:
        if self._duration_sec is not None:
            return self._duration_sec
        return float(len(self.waveform)) / float(self.sample_rate) if self.sample_rate > 0 else 0.0

    @property
    def num_samples(self) -> int:
        return len(self.waveform)

    def to_wav_bytes(self) -> bytes:
        """
        Convert AudioData back to a PCM 16-bit mono WAV bytestring
        with a proper RIFF/WAVE header.

        This is the inverse of ``_decode_wav`` — useful when an audio
        loaded via ID-based fetch needs to re-enter the bytes path
        (e.g. for ``verify_from_bytes``).
        """
        import struct
        pcm = (self.waveform * 32767.0).clip(-32768, 32767).astype(np.int16)
        data = pcm.tobytes()
        data_size = len(data)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,  # total file size - 8
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,  # PCM format
            1,  # mono
            self.sample_rate,
            self.sample_rate * 2,  # byte rate
            2,  # block align
            16,  # bits per sample
            b"data",
            data_size,
        )
        return header + data


# ---------------------------------------------------------------------------
# Audio loader
# ---------------------------------------------------------------------------
class AudioLoader:
    """
    Loads audio from various sources and preprocesses it for speaker verification.

    Supports two modes:
    1. Direct: decode raw bytes (from multipart upload)
    2. Indirect: fetch audio by ID from NAS / S3 / Redis
    """

    def __init__(
        self,
        audio_config: AudioConfig,
        storage_config: StorageConfig,
        fetcher: Optional[AudioFetcher] = None,
    ) -> None:
        self._audio_config = audio_config
        self._storage_config = storage_config
        self._fetcher = fetcher
        self._target_sr = audio_config.target_sample_rate

    # ------------------------------------------------------------------
    # Direct mode — decode from raw bytes
    # ------------------------------------------------------------------

    def load_from_bytes(
        self,
        audio_bytes: bytes,
        audio_id: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> AudioData:
        """
        Decode raw audio bytes into an AudioData object.

        Supports WAV (header-based), raw PCM, and ulaw/alaw.
        For MP3/FLAC/OGG, requires soundfile/ffmpeg (falls back to error message).
        """
        if not audio_bytes or len(audio_bytes) < 44:
            raise AudioLoadError(f"Audio too short ({len(audio_bytes)} bytes)")

        fmt_lower = (fmt or "wav").lower()

        if fmt_lower == "wav" and audio_bytes[:4] == b"RIFF":
            return self._decode_wav(audio_bytes, audio_id)
        elif fmt_lower in ("ulaw", "ulaw"):
            return self._decode_ulaw(audio_bytes, audio_id)
        elif fmt_lower in ("alaw", "pcm_alaw"):
            return self._decode_alaw(audio_bytes, audio_id)
        elif fmt_lower == "raw":
            return self._decode_raw_pcm(audio_bytes, audio_id)
        else:
            # Try soundfile if available, else raise
            try:
                return self._decode_with_soundfile(audio_bytes, audio_id)
            except ImportError:
                raise AudioLoadError(
                    f"Format '{fmt_lower}' requires `soundfile` or `ffmpeg`; "
                    "install with: pip install soundfile"
                )

    def load_from_file(self, filepath: str, audio_id: Optional[str] = None) -> AudioData:
        """Load audio from a local file path."""
        path = Path(filepath)
        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {filepath}")
        audio_bytes = path.read_bytes()
        return self.load_from_bytes(
            audio_bytes,
            audio_id=audio_id or path.stem,
            fmt=path.suffix.lstrip("."),
        )

    # ------------------------------------------------------------------
    # URL download mode
    # ------------------------------------------------------------------

    def download_url(self, url: str) -> bytes:
        """
        Download audio from a URL, cache to local disk, return raw bytes.

        The local file is named ``<md5(url)><ext>`` and stored under
        ``storage.download_dir`` (default: ``<api>/downloads/``).
        Subsequent calls with the same URL skip the download.

        Returns:
            Raw audio bytes (suitable for ``verify_from_bytes``).

        Raises:
            AudioLoadError: If download fails.
        """
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

        # Resolve download directory
        raw_dir = self._storage_config.download_dir
        if raw_dir:
            download_dir = Path(raw_dir)
        else:
            # Fallback: <api>/downloads/
            download_dir = Path(__file__).resolve().parent.parent / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        # --- Check local cache -------------------------------------------------
        known_exts = [".wav", ".ulaw", ".alaw", ".mp3", ".flac", ".ogg", ".bin"]
        for ext in known_exts:
            cached = download_dir / f"{url_hash}{ext}"
            if cached.exists():
                logger.debug("URL cache HIT: %s -> %s", url, cached)
                return cached.read_bytes()

        # --- Download -----------------------------------------------------------
        logger.info("Downloading audio from URL: %s", url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ASV-API/1.0 (SpeakerVerification)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio_bytes = resp.read()
        except Exception as e:
            raise AudioLoadError(f"Failed to download audio from '{url}': {e}")

        if not audio_bytes or len(audio_bytes) < 44:
            raise AudioLoadError(
                f"Downloaded audio from '{url}' is too small ({len(audio_bytes)} bytes)"
            )

        # --- Determine file extension ------------------------------------------
        ext = self._infer_ext_from_url(url)
        if not ext:
            ext = self._infer_ext_from_bytes(audio_bytes)
        if not ext:
            ext = ".wav"  # safe default

        save_path = download_dir / f"{url_hash}{ext}"
        save_path.write_bytes(audio_bytes)
        logger.info("URL download saved: %s -> %s (%d bytes)", url, save_path, len(audio_bytes))

        return audio_bytes

    @staticmethod
    def _infer_ext_from_url(url: str) -> Optional[str]:
        """Extract known audio extension from a URL path."""
        from urllib.parse import urlparse
        path = urlparse(url).path
        ext = Path(path).suffix.lower()
        known = {".wav", ".ulaw", ".alaw", ".mp3", ".flac", ".ogg", ".raw", ".pcm"}
        return ext if ext in known else None

    @staticmethod
    def _infer_ext_from_bytes(data: bytes) -> Optional[str]:
        """Sniff audio extension from header magic bytes."""
        if len(data) < 4:
            return None
        if data[:4] == b"RIFF":
            return ".wav"
        if data[:4] == b"fLaC":
            return ".flac"
        if data[:3] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or data[:2] in (
            b"\xff\xfb", b"\xff\xf3", b"\xff\xf2",
        ):
            return ".mp3"
        if data[:4] == b"OggS":
            return ".ogg"
        return None

    # ------------------------------------------------------------------
    # Indirect mode — fetch by ID
    # ------------------------------------------------------------------

    def load_by_id(
        self,
        audio_id: str,
        backend: str = "local_file",
        bucket: Optional[str] = None,
    ) -> AudioData:
        """
        Fetch audio by ID using the configured fetcher plugin.

        Args:
            audio_id: Unique audio identifier (e.g., recording UUID).
            backend: Ignored when ``self._fetcher`` is set (the fetcher
                     type is already configured).  Kept for backward-compat
                     with the old ``verify_from_ids`` signature.
            bucket: Per-request bucket override for S3 fetcher.

        Returns:
            AudioData object.

        Raises:
            AudioLoadError: If audio cannot be found or decoded.
        """
        if self._fetcher is not None:
            # Modern path: use the configured fetcher plugin
            try:
                kwargs = {}
                if bucket is not None:
                    kwargs["bucket"] = bucket
                raw_bytes = self._fetcher.fetch(audio_id, **kwargs)
            except FetchError as e:
                raise AudioLoadError(str(e)) from e
        else:
            # Legacy fallback: use StorageConfig-based backends
            raw_bytes = self._legacy_fetch(audio_id, backend, bucket or "")

        return self.load_from_bytes(raw_bytes, audio_id=audio_id)

    def _legacy_fetch(self, audio_id: str, backend: str, bucket: str) -> bytes:
        """Fallback: original hardcoded backend dispatch."""
        if backend == "nas" or backend == "local_file":
            return self._load_bytes_from_nas(audio_id)
        elif backend == "s3":
            return self._load_bytes_from_s3(audio_id, bucket)
        elif backend == "redis":
            return self._load_bytes_from_redis(audio_id)
        else:
            raise AudioLoadError(f"Unknown storage backend: {backend}")

    def _load_bytes_from_nas(self, audio_id: str) -> bytes:
        """Load raw audio bytes from NAS mount path."""
        base = Path(self._storage_config.nas_mount_path)
        for ext in [".wav", ".ulaw", ".alaw", ".mp3", ".flac"]:
            candidate = base / f"{audio_id}{ext}"
            if candidate.exists():
                return candidate.read_bytes()
            for sub in candidate.parent.rglob(f"{audio_id}{ext}"):
                return sub.read_bytes()
        raise AudioLoadError(
            f"Audio '{audio_id}' not found in {base} "
            f"(tried: .wav, .ulaw, .alaw, .mp3, .flac)"
        )

    def _load_bytes_from_s3(self, audio_id: str, bucket: str) -> bytes:
        """Load raw audio bytes from S3."""
        try:
            import boto3
        except ImportError:
            raise AudioLoadError("boto3 not installed; pip install boto3")
        if not bucket:
            bucket = self._storage_config.s3_bucket
        if not bucket:
            raise AudioLoadError("S3 bucket not configured")
        s3 = boto3.client(
            "s3",
            endpoint_url=self._storage_config.s3_endpoint or None,
            aws_access_key_id=self._storage_config.s3_access_key or None,
            aws_secret_access_key=self._storage_config.s3_secret_key or None,
        )
        try:
            response = s3.get_object(Bucket=bucket, Key=audio_id)
            return response["Body"].read()
        except Exception as e:
            raise AudioLoadError(f"S3 fetch failed for '{audio_id}': {e}")

    def _load_bytes_from_redis(self, audio_id: str) -> bytes:
        """Load raw audio bytes from Redis."""
        try:
            import redis as redis_client
        except ImportError:
            raise AudioLoadError("redis not installed; pip install redis")
        r = redis_client.from_url("redis://localhost:6379/0")

    def preprocess(self, audio: AudioData) -> AudioData:
        """
        Full preprocessing pipeline for speaker verification:
        1. Resample to target sample rate
        2. Apply VAD (remove silence)
        3. Validate duration
        4. Extract fbank features (if needed downstream)

        Returns preprocessed AudioData (waveform unchanged except resampling).
        """
        # 1. Resample
        waveform = self._resample(audio.waveform, audio.sample_rate, self._target_sr)

        # 2. VAD
        vad_config = self._audio_config
        valid_mask = self._apply_vad(
            waveform,
            self._target_sr,
            window_ms=vad_config.vad_window_ms,
            threshold=vad_config.vad_threshold,
        ) if vad_config.vad_enabled else None

        if valid_mask is not None:
            valid_sec = float(np.sum(valid_mask)) / float(self._target_sr)
        else:
            valid_sec = float(len(waveform)) / float(self._target_sr)

        # 3. Duration check
        min_dur = self._audio_config.min_duration_sec
        if valid_sec < min_dur:
            raise InsufficientAudioError(
                f"Valid speech duration {valid_sec:.2f}s < minimum {min_dur:.2f}s"
            )

        max_dur = self._audio_config.max_duration_sec
        if len(waveform) > max_dur * self._target_sr:
            logger.warning(
                "Audio %.2fs exceeds max %.2fs, truncating",
                len(waveform) / self._target_sr,
                max_dur,
            )
            waveform = waveform[: int(max_dur * self._target_sr)]

        result = AudioData(
            waveform=waveform,
            sample_rate=self._target_sr,
            audio_id=audio.audio_id,
            duration_sec=len(waveform) / self._target_sr,
        )
        return result

    def extract_fbank(
        self,
        audio: AudioData,
        num_filters: int = 80,
        window_ms: float = 25.0,
        hop_ms: float = 10.0,
    ) -> np.ndarray:
        """
        Extract log-mel fbank features from audio waveform.

        Returns numpy array of shape (num_frames, num_filters).
        This is a lightweight implementation using numpy FFT.
        For production, consider using torchaudio.compliance.kaldi.fbank
        or python_speech_features.
        """
        sr = audio.sample_rate
        waveform = audio.waveform

        win_length = int(sr * window_ms / 1000.0)
        hop_length = int(sr * hop_ms / 1000.0)
        n_fft = 1
        while n_fft < win_length:
            n_fft <<= 1

        # Pre-emphasis
        emphasized = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

        # Framing
        num_frames = 1 + (len(emphasized) - win_length) // hop_length
        if num_frames < 1:
            # Pad if too short
            pad_len = win_length - len(emphasized)
            emphasized = np.pad(emphasized, (0, max(0, pad_len)))
            num_frames = 1

        frames = np.lib.stride_tricks.sliding_window_view(
            emphasized, win_length
        )[::hop_length]
        if frames.ndim == 1:
            frames = frames[np.newaxis, :]

        # Window (Hamming)
        window = np.hamming(win_length)
        frames = frames * window

        # FFT -> power spectrum
        magnitude = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

        # Mel filterbank
        mel_w = self._mel_filterbank(
            num_filters, n_fft, sr, f_min=0, f_max=sr / 2
        )
        mel_energy = magnitude @ mel_w

        # Log
        mel_energy = np.maximum(mel_energy, 1e-10)
        log_mel = np.log(mel_energy)

        # Mean-variance normalization (per utterance cepstral mean)
        log_mel = (log_mel - np.mean(log_mel, axis=1, keepdims=True)) / (
            np.std(log_mel, axis=1, keepdims=True) + 1e-10
        )

        return log_mel.astype(np.float32)

    # ------------------------------------------------------------------
    # Decoders
    # ------------------------------------------------------------------

    def _decode_wav(self, raw: bytes, audio_id: Optional[str] = None) -> AudioData:
        """Decode WAV file from raw bytes. Handles common PCM formats."""
        try:
            sample_rate, waveform = self._wav_bytes_to_numpy(raw)
            return AudioData(waveform=waveform, sample_rate=sample_rate, audio_id=audio_id)
        except Exception as e:
            raise AudioLoadError(f"Failed to decode WAV: {e}")

    def _decode_ulaw(self, raw: bytes, audio_id: Optional[str] = None) -> AudioData:
        """Decode μ-law encoded audio (common in telephone recordings, 8kHz)."""
        # μ-law to linear PCM conversion
        ulaw_table = self._build_ulaw_table()
        pcm = np.array([ulaw_table[b & 0xFF] for b in raw], dtype=np.int16)
        waveform = pcm.astype(np.float32) / 32768.0
        return AudioData(waveform=waveform, sample_rate=8000, audio_id=audio_id)

    def _decode_alaw(self, raw: bytes, audio_id: Optional[str] = None) -> AudioData:
        """Decode A-law encoded audio."""
        alaw_table = self._build_alaw_table()
        pcm = np.array([alaw_table[b & 0xFF] for b in raw], dtype=np.int16)
        waveform = pcm.astype(np.float32) / 32768.0
        return AudioData(waveform=waveform, sample_rate=8000, audio_id=audio_id)

    def _decode_raw_pcm(self, raw: bytes, audio_id: Optional[str] = None) -> AudioData:
        """Decode raw 16-bit PCM (no header). Assumes 8kHz mono."""
        pcm = np.frombuffer(raw, dtype=np.int16)
        waveform = pcm.astype(np.float32) / 32768.0
        return AudioData(waveform=waveform, sample_rate=8000, audio_id=audio_id)

    def _decode_with_soundfile(self, raw: bytes, audio_id: Optional[str] = None) -> AudioData:
        """Decode using soundfile library (supports FLAC, OGG, MP3 via ffmpeg)."""
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono mixdown
        return AudioData(waveform=data, sample_rate=sr, audio_id=audio_id)

    # ------------------------------------------------------------------
    # DSP utilities
    # ------------------------------------------------------------------

    def _resample(self, waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resample waveform to target sample rate using linear interpolation."""
        if src_sr == dst_sr:
            return waveform

        src_len = len(waveform)
        dst_len = int(round(float(src_len) * float(dst_sr) / float(src_sr)))
        indices = np.linspace(0, src_len - 1, dst_len)
        return np.interp(indices, np.arange(src_len), waveform).astype(np.float32)

    def _apply_vad(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        window_ms: int = 30,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Simple energy-based VAD. Returns boolean mask per window.

        Args:
            waveform: Audio waveform (float32 [-1, 1]).
            sample_rate: Sample rate in Hz.
            window_ms: Window size in ms.
            threshold: Energy threshold (0-1, relative to max energy).

        Returns:
            Boolean array where True indicates active speech.
        """
        win_len = int(sample_rate * window_ms / 1000)
        if win_len < 1:
            win_len = 1
        num_windows = len(waveform) // win_len
        if num_windows < 1:
            return np.zeros(0, dtype=bool)

        # Compute RMS energy per window
        windows = waveform[: num_windows * win_len].reshape(num_windows, win_len)
        energies = np.sqrt(np.mean(windows ** 2, axis=1))

        # Adaptive threshold (relative to max)
        energy_max = energies.max()
        if energy_max < 1e-10:
            return np.zeros(num_windows, dtype=bool)
        threshold_abs = energy_max * threshold

        return energies >= threshold_abs

    def _mel_filterbank(
        self,
        n_mels: int,
        n_fft: int,
        sr: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> np.ndarray:
        """Compute mel filterbank matrix (simplified implementation)."""
        if f_max is None:
            f_max = sr / 2.0

        # Convert Hz to mel scale
        def hz_to_mel(f: float) -> float:
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        # Mel points
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mels])

        # FFT bin frequencies
        fft_bins = (n_fft // 2) + 1
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fb = np.zeros((n_mels, fft_bins))

        for i in range(n_mels):
            left = bins[i]
            center = bins[i + 1]
            right = bins[i + 2]
            if center > left:
                fb[i, left:center] = np.linspace(0, 1, center - left)
            if right > center:
                fb[i, center:right] = np.linspace(1, 0, right - center)

        return fb.T

    # ------------------------------------------------------------------
    # μ-law / A-law conversion tables
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ulaw_table() -> List[int]:
        """Build μ-law decoding lookup table (8-bit to 16-bit PCM)."""
        table = []
        BIAS = 132
        for i in range(256):
            mantissa = i & 0x0F
            exponent = (i >> 4) & 0x07
            sign = 1 if (i & 0x80) else -1
            magnitude = (mantissa << 1) + 1 + BIAS
            magnitude = magnitude << (exponent + 2)
            table.append(sign * magnitude)
        return table

    @staticmethod
    def _build_alaw_table() -> List[int]:
        """Build A-law decoding lookup table (8-bit to 16-bit PCM)."""
        table = []
        for i in range(256):
            input_val = i ^ 0x55  # XOR with 0x55
            sign = 1 if (input_val & 0x80) else -1
            exponent = (input_val >> 4) & 0x07
            mantissa = input_val & 0x0F
            if exponent > 0:
                magnitude = (mantissa | 0x10) << (exponent + 3)
            else:
                magnitude = (mantissa << 1) | 1
            table.append(sign * magnitude)
        return table

    @staticmethod
    def _wav_bytes_to_numpy(wav_bytes: bytes) -> Tuple[int, np.ndarray]:
        """
        Parse WAV bytes and return (sample_rate, waveform_float32).
        Handles PCM 8/16/24/32-bit.
        """
        import struct

        if wav_bytes[:4] != b"RIFF":
            raise ValueError("Not a RIFF file")
        if wav_bytes[8:12] != b"WAVE":
            raise ValueError("Not a WAVE file")

        # Find fmt chunk
        pos = 12
        sample_rate = 8000
        bits_per_sample = 16
        num_channels = 1
        data_bytes = b""

        while pos < len(wav_bytes) - 8:
            chunk_id = wav_bytes[pos:pos+4]
            chunk_size = struct.unpack("<I", wav_bytes[pos+4:pos+8])[0]
            if chunk_id == b"fmt ":
                fmt = wav_bytes[pos+8:pos+8+chunk_size]  # typically 16 bytes for PCM
                audio_format, num_channels, sample_rate, _, _, bits_per_sample = \
                    struct.unpack("<HHIIHH", fmt)
                if audio_format != 1:  # PCM
                    raise ValueError(f"Unsupported audio format: {audio_format}")
            elif chunk_id == b"data":
                data_bytes = wav_bytes[pos+8:pos+8+chunk_size]
            pos += 8 + chunk_size
            if chunk_id == b"data":
                break

        if not data_bytes:
            raise ValueError("No data chunk found")

        # Decode PCM
        if bits_per_sample == 8:
            dtype = np.uint8
            raw = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
            waveform = (raw - 128.0) / 128.0
        elif bits_per_sample == 16:
            raw = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32)
            waveform = raw / 32768.0
        elif bits_per_sample == 24:
            raw = np.frombuffer(data_bytes, dtype=np.uint8).reshape(-1, 3)
            pcm = np.zeros(len(raw), dtype=np.int32)
            for i in range(len(raw)):
                val = int.from_bytes(raw[i].tobytes(), "little", signed=True)
                pcm[i] = val
            waveform = pcm.astype(np.float32) / 8388608.0
        elif bits_per_sample == 32:
            raw = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float32)
            waveform = raw / 2147483648.0
        else:
            raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

        # Mix down to mono
        if num_channels > 1:
            waveform = waveform.reshape(-1, num_channels).mean(axis=1)

        return sample_rate, waveform


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class AudioLoadError(Exception):
    """Raised when audio cannot be loaded or decoded."""
    pass


class InsufficientAudioError(AudioLoadError):
    """Raised when audio does not have enough valid speech."""
    CODE = "INSUFFICIENT_AUDIO"

    def __init__(self, message: str = "Insufficient valid speech duration") -> None:
        super().__init__(message)
        self.code = self.CODE
