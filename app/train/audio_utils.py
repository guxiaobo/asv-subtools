"""
音频工具函数。

提供录音加载、重采样、降噪、SNR 计算等基础操作。
尽量使用 numpy + wave 实现，减少外部依赖。
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("train.audio_utils")

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_wave(path: str) -> Tuple[np.ndarray, int]:
    """
    使用标准库 wave 模块加载 WAV 文件。

    Args:
        path: WAV 文件路径。

    Returns:
        (waveform, sample_rate)
        waveform: float32, [-1, 1], shape (num_samples,)
        sample_rate: int

    Raises:
        FileNotFoundError, ValueError: 文件不存在或格式不支持。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Convert raw bytes to numpy
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Normalize and handle multiple channels (take first channel)
    if sampwidth == 1:
        audio = (audio - 128.0) / 128.0
    else:
        scale = {2: 32768.0, 4: 2147483648.0}.get(sampwidth, 32768.0)
        audio = audio / scale

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    return audio, sr


def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    加载音频文件，可选重采样。

    优先使用 wave (WAV)，尝试 soundfile 支持更多格式，
    最终回退到 ffmpeg (AMR/AMR-WB/OPUS 等)。

    Args:
        path: 音频文件路径。
        target_sr: 目标采样率，None 则保持原始采样率。

    Returns:
        (waveform, sample_rate)
    """
    path_lower = path.lower()

    errors = []

    # Try stdlib wave first for WAV files
    if path_lower.endswith(".wav"):
        try:
            waveform, sr = load_wave(path)
            return _maybe_resample(waveform, sr, target_sr)
        except Exception as e:
            errors.append(f"wave: {e}")
            logger.warning("wave load failed for %s: %s, trying alternatives", path, e)

    # Try soundfile
    try:
        waveform, sr = _load_with_soundfile(path)
        return _maybe_resample(waveform, sr, target_sr)
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # Final fallback: ffmpeg
    try:
        effective_sr = target_sr if target_sr is not None else 16000
        waveform, sr = _load_with_ffmpeg(path, effective_sr)
        return _maybe_resample(waveform, sr, target_sr)
    except Exception as e:
        errors.append(f"ffmpeg: {e}")

    raise RuntimeError(
        f"Cannot load audio '{path}'. All backends failed:\n" + "\n".join(errors)
    )


def _maybe_resample(
    waveform: np.ndarray, sr: int, target_sr: Optional[int]
) -> Tuple[np.ndarray, int]:
    """如果需要则重采样，否则原样返回。"""
    if target_sr is not None and target_sr != sr:
        waveform = resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def _load_with_soundfile(path: str) -> Tuple[np.ndarray, int]:
    """使用 soundfile 加载音频（支持 WAV, FLAC, OGG 等）。"""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1 and data.shape[1] > 1:
        data = data[:, 0]
    return data, sr


def _load_with_ffmpeg(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """使用 ffmpeg 加载任意格式音频（AMR/AMR-WB/OPUS 等）。"""
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-acodec", "pcm_s16le",
        "-ac", "1",            # mono
        "-ar", str(target_sr),
        "-f", "wav",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found; install ffmpeg: brew install ffmpeg")
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code={result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')[:300]}"
        )
    from io import BytesIO
    wav_buf = BytesIO(result.stdout)
    with wave.open(wav_buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    logger.info("ffmpeg decoded %s -> %d samples @ %dHz", path, len(data), sr)
    return data, sr


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    对音频进行重采样。

    使用 numpy 实现的线性插值重采样，避免依赖 librosa。

    Args:
        waveform: 输入波形 (float32, [-1, 1])。
        orig_sr: 原始采样率。
        target_sr: 目标采样率。

    Returns:
        重采样后的波形。
    """
    if orig_sr == target_sr:
        return waveform

    n_samples = len(waveform)
    duration = n_samples / orig_sr
    target_n = int(duration * target_sr)

    # Linear interpolation
    x_old = np.linspace(0, n_samples - 1, n_samples)
    x_new = np.linspace(0, n_samples - 1, target_n)
    resampled = np.interp(x_new, x_old, waveform)

    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------


def save_wav(path: str, waveform: np.ndarray, sample_rate: int) -> str:
    """
    将 numpy 波形保存为 WAV 文件。

    Args:
        path: 保存路径。
        waveform: float32, [-1, 1]。
        sample_rate: 采样率。

    Returns:
        绝对路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Clip and convert to int16
    waveform_clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform_clipped * 32767.0).astype(np.int16)

    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return str(p.resolve())


def save_segments(
    output_dir: str,
    prefix: str,
    segments: List[Tuple[np.ndarray, int]],
    sample_rate: int,
) -> List[str]:
    """
    保存多个音频段到指定目录。

    Args:
        output_dir: 输出目录。
        prefix: 文件名前缀。
        segments: (波形, 编号) 列表。
        sample_rate: 采样率。

    Returns:
        保存的文件路径列表。
    """
    paths = []
    for i, (wav, _) in enumerate(segments):
        seg_path = Path(output_dir) / f"{prefix}_seg{i+1:03d}.wav"
        save_wav(str(seg_path), wav, sample_rate)
        paths.append(str(seg_path))
    return paths


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


def estimate_snr(waveform: np.ndarray, sample_rate: int) -> float:
    """
    估计音频信噪比 (SNR)。

    使用简单的能量比：将低能量帧视为噪声，高能量帧视为信号。

    Args:
        waveform: 输入波形。
        sample_rate: 采样率。

    Returns:
        SNR 值（dB）。
    """
    if len(waveform) == 0:
        return 0.0

    # Frame-based energy computation
    frame_len = int(0.025 * sample_rate)  # 25ms
    hop_len = int(0.010 * sample_rate)  # 10ms
    energy_list = []

    for start in range(0, len(waveform) - frame_len, hop_len):
        frame = waveform[start: start + frame_len]
        energy = np.mean(frame ** 2) + 1e-10
        energy_list.append(energy)

    if not energy_list:
        return 0.0

    energies = np.array(energy_list)

    # Use top/bottom percentile as signal/noise
    signal_energy = np.percentile(energies, 90)
    noise_energy = np.percentile(energies, 10)

    if noise_energy <= 0:
        return 40.0  # No noise detected

    snr = 10.0 * np.log10(signal_energy / noise_energy)
    return float(np.clip(snr, -10, 60))


# ---------------------------------------------------------------------------
# Simple noise reduction (spectral gating)
# ---------------------------------------------------------------------------


def reduce_noise(
    waveform: np.ndarray,
    sample_rate: int,
    noise_reduction_strength: float = 0.8,
) -> np.ndarray:
    """
    简单的噪声抑制。

    使用基于帧的频谱门控（spectral gating）方法。
    如果没有安装 noisereduce 库，使用简化版本。

    Args:
        waveform: 输入波形。
        sample_rate: 采样率。
        noise_reduction_strength: 降噪强度 (0-1)。

    Returns:
        降噪后的波形。
    """
    try:
        import noisereduce as nr
        # Use stationary noise reduction (noise profile from first 500ms)
        noise_sample = waveform[:min(int(0.5 * sample_rate), len(waveform))]
        return nr.reduce_noise(
            y=waveform,
            sr=sample_rate,
            y_noise=noise_sample,
            prop_decrease=noise_reduction_strength,
        )
    except ImportError:
        # Fallback: simple energy-based noise gate
        return _simple_noise_gate(waveform, sample_rate)


def _simple_noise_gate(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """简易噪声门（能量低于阈值则静音）。"""
    frame_len = int(0.025 * sample_rate)
    if frame_len <= 0:
        return waveform

    result = waveform.copy()

    for start in range(0, len(waveform), frame_len):
        end = min(start + frame_len, len(waveform))
        frame = waveform[start:end]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        if rms < 0.01:  # -40dB threshold
            result[start:end] = 0.0

    return result
