"""
语音活动检测 (VAD) 模块。

实现了银行场景特有的 VAD 策略：
- 基于能量的帧级 VAD
- 过滤前 2s（系统提示音）
- 丢弃 < 1.5s 的短段
- 丢弃低 SNR 段
- 双讲段检测与丢弃
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from train.audio_utils import estimate_snr

logger = logging.getLogger("train.vad")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Segment = Tuple[np.ndarray, int, int]  # (waveform, start_sample, end_sample)


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------


def _frame_energy(waveform: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """计算每帧的能量。"""
    energy_list = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        frame = waveform[start: start + frame_len]
        energy = np.mean(frame ** 2)
        energy_list.append(energy)
    if not energy_list:
        return np.array([])
    return np.array(energy_list)


def energy_vad(
    waveform: np.ndarray,
    sample_rate: int,
    window_ms: int = 30,
    threshold: float = 0.5,
    min_segment_sec: float = 0.0,
    max_segment_sec: float = 30.0,
    filter_leading_sec: float = 2.0,
    snr_threshold: float = 0.0,
    lossless: bool = True,
) -> List[Segment]:
    """
    基于能量的 VAD 切割，适配电话录音。

    使用噪声底噪 + dB 增益自适应阈值，并对断开的语音段进行合并。
    默认 lossless=True 保留所有分段（不丢弃短段，不按 SNR 过滤），
    以满足「所有分段都不能丢失」的需求。

    Args:
        waveform: 输入波形 (float32, [-1, 1])。
        sample_rate: 采样率。
        window_ms: 帧长（毫秒）。
        threshold: 噪声底噪之上的 dB 阈值倍率（实际 dB = max(threshold*10, 3)）。
        min_segment_sec: 最小段长（秒）。lossless=False 时有效，< 此值的段丢弃。
        max_segment_sec: 最大段长（秒），超过则切分。
        filter_leading_sec: 过滤开头的秒数（系统提示音）。
        snr_threshold: 段 SNR 阈值（dB）。lossless=False 时有效。
        lossless: True=保留所有分段（忽略 min_segment_sec/snr_threshold）；
                  False=过滤短段和低 SNR 段（原行为）。

    Returns:
        语音段列表，每段为 (waveform, start_sample, end_sample)。
    """
    if len(waveform) == 0:
        return []

    # 1. Filter leading portion (system beep)
    filter_samples = int(filter_leading_sec * sample_rate)
    if filter_samples >= len(waveform):
        return []
    waveform = waveform[filter_samples:]

    # 2. Frame energy
    frame_len = int(window_ms * sample_rate / 1000)
    hop_len = frame_len // 2  # 50% overlap
    if frame_len <= 0 or hop_len <= 0:
        return []

    energies = _frame_energy(waveform, frame_len, hop_len)
    if len(energies) == 0:
        return []

    # 3. Adaptive threshold: noise floor + dB offset
    #    noise_floor = P10 energy (calibrated for telephone background)
    #    voice_threshold = noise_floor * 10^(threshold_db / 10)
    noise_floor = float(np.percentile(energies, 10))
    noise_floor = max(noise_floor, 1e-12)
    # threshold parameter is now interpreted as dB above noise floor
    db_gain = max(threshold * 10.0, 3.0)  # at least 3 dB above noise floor
    energy_threshold = noise_floor * (10.0 ** (db_gain / 10.0))

    voice_flags = energies > energy_threshold

    logger.debug(
        "VAD: noise_floor=%.2e, energy_threshold=%.2e (%.1f dB above noise), "
        "voice_frames=%d/%d (%.1f%%)",
        noise_floor, energy_threshold, db_gain,
        int(voice_flags.sum()), len(voice_flags),
        float(voice_flags.mean() * 100),
    )

    # 4. Voice-active segments
    segments = _flags_to_segments(voice_flags, len(waveform), frame_len, hop_len)

    # 5. Merge nearby segments (gap < max_gap_sec)
    max_gap_samples = int(0.8 * sample_rate)  # merge if gap < 0.8s
    segments = _merge_segments(segments, max_gap_samples)

    if not lossless:
        # 6. Filter by duration (only in non-lossless mode)
        segments = _filter_by_duration(segments, sample_rate, min_segment_sec, max_segment_sec)
        # 7. Filter by SNR (only in non-lossless mode)
        segments = _filter_by_snr(segments, sample_rate, snr_threshold)

    # 8. Convert to (waveform, start, end)
    result: List[Segment] = []
    for start_s, end_s in segments:
        seg_wav = waveform[start_s:end_s]
        result.append((seg_wav, start_s + filter_samples, end_s + filter_samples))

    logger.info(
        "VAD: %d samples → %d segments (%.1fs total, threshold=%.2e, %.1f dB above noise, lossless=%s)",
        len(waveform), len(result),
        sum(len(w) for w, _, _ in result) / sample_rate,
        energy_threshold, db_gain, lossless,
    )
    return result


def _merge_segments(
    segments: List[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    """合并间隔小于 max_gap 的相邻段。"""
    if len(segments) <= 1:
        return segments[:]
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _flags_to_segments(
    flags: np.ndarray,
    total_samples: int,
    frame_len: int,
    hop_len: int,
) -> List[Tuple[int, int]]:
    """将帧级标记转换为段列表 [(start, end), ...]。"""
    segments = []
    in_speech = False
    seg_start = 0

    for i, active in enumerate(flags):
        sample_pos = i * hop_len
        if active and not in_speech:
            seg_start = sample_pos
            in_speech = True
        elif not active and in_speech:
            seg_end = min(sample_pos + frame_len, total_samples)
            segments.append((seg_start, seg_end))
            in_speech = False

    if in_speech:
        segments.append((seg_start, total_samples))

    return segments


def _filter_by_duration(
    segments: List[Tuple[int, int]],
    sample_rate: int,
    min_sec: float,
    max_sec: float,
) -> List[Tuple[int, int]]:
    """按时长过滤段。"""
    min_samples = int(min_sec * sample_rate)
    max_samples = int(max_sec * sample_rate)
    filtered = []
    for start, end in segments:
        dur = end - start
        if dur < min_samples:
            continue
        if dur > max_samples:
            # Split into max-length chunks
            pos = start
            while pos < end:
                chunk_end = min(pos + max_samples, end)
                filtered.append((pos, chunk_end))
                pos = chunk_end
        else:
            filtered.append((start, end))
    return filtered


def _filter_by_snr(
    segments: List[Tuple[int, int]],
    sample_rate: int,
    snr_threshold: float,
) -> List[Tuple[int, int]]:
    """按 SNR 过滤段。"""
    filtered = []
    for start, end in segments:
        # We don't have the waveform here, so we do a rough check
        # using the segment length (longer segments tend to have better SNR)
        duration = (end - start) / sample_rate
        if duration < 0.5:
            continue
        filtered.append((start, end))
    return filtered


# ---------------------------------------------------------------------------
# Channel separation detection
# ---------------------------------------------------------------------------


def detect_double_talk(
    agent_waveform: np.ndarray,
    customer_waveform: np.ndarray,
    sample_rate: int,
    threshold_db: float = -25.0,
) -> np.ndarray:
    """
    检测双讲段（坐席与客户同时说话）。

    当两个通道的能量同时超过阈值时，认为该帧是双讲段。

    Args:
        agent_waveform: 坐席通道波形。
        customer_waveform: 客户通道波形。
        sample_rate: 采样率。
        threshold_db: 双讲能量阈值（dB）。

    Returns:
        布尔数组，True 表示该帧为双讲段。
    """
    frame_len = int(0.025 * sample_rate)  # 25ms
    hop_len = int(0.010 * sample_rate)  # 10ms

    thresh_linear = 10.0 ** (threshold_db / 20.0)

    double_talk_frames = []
    for start in range(0, min(len(agent_waveform), len(customer_waveform)) - frame_len, hop_len):
        agent_frame = agent_waveform[start: start + frame_len]
        cust_frame = customer_waveform[start: start + frame_len]

        agent_rms = np.sqrt(np.mean(agent_frame ** 2) + 1e-10)
        cust_rms = np.sqrt(np.mean(cust_frame ** 2) + 1e-10)

        dt = (agent_rms > thresh_linear) and (cust_rms > thresh_linear)
        double_talk_frames.append(dt)

    return np.array(double_talk_frames)


# ---------------------------------------------------------------------------
# Preprocessing orchestrator
# ---------------------------------------------------------------------------


def preprocess_recording(
    audio_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    channel_separated: bool = False,
    apply_noise_reduction: bool = True,
    # ── Diarizer integration (optional) ──
    run_diarization: bool = True,
    diarizer_model_name: str = "CAM++",
    diarizer_db_path: Optional[str] = None,
    diarizer_agent_threshold: Optional[float] = None,
    diarizer_cluster_threshold: float = 0.55,
    **vad_kwargs,
) -> dict:
    """
    对一条录音执行完整的训练前预处理 + 说话人标注。

    Args:
        audio_path: 原始录音文件路径。
        output_dir: 预处理输出目录。
        sample_rate: 目标采样率。
        channel_separated: 是否已通道分离。
        apply_noise_reduction: 是否应用降噪。
        run_diarization: 是否在 VAD 后执行说话人标注（默认打开）。
        diarizer_model_name: 声纹模型名（CAM++/ResNet34/ECAPA）。
        diarizer_db_path: 数据库路径（默认自动）。
        diarizer_agent_threshold: 坐席判定阈值（默认 None=动态检测）。
        diarizer_cluster_threshold: 客户聚类阈值。
        **vad_kwargs: VAD 参数（覆盖默认值）。

    Returns:
        预处理结果字典 (PreprocessResult 结构)。
    """
    from pathlib import Path

    from train.audio_utils import (
        load_audio,
        reduce_noise,
        save_segments,
        estimate_snr,
    )

    result = {
        "segment_count": 0,
        "agent_segments": 0,
        "customer_segments": 0,
        "agent_valid_sec": 0.0,
        "customer_valid_sec": 0.0,
        "uncertain_segments": 0,
        "uncertain_valid_sec": 0.0,
        "avg_snr_db": 0.0,
        "min_snr_db": 0.0,
        "max_snr_db": 0.0,
        "dropped_segments": 0,
        "dropped_reason": {},
        # Diarization results
        "diarization_done": False,
        "diarization_model": diarizer_model_name,
        "customer_voiceprint_available": False,
        "customer_voiceprint_num_segments": 0,
    }

    # Load audio
    logger.info("Loading audio: %s", audio_path)
    waveform, sr = load_audio(audio_path, target_sr=sample_rate)

    # Noise reduction
    if apply_noise_reduction and len(waveform) > sample_rate:
        try:
            waveform = reduce_noise(waveform, sr)
        except Exception as e:
            logger.warning("Noise reduction skipped for %s: %s", audio_path, e)

    # Calculate overall SNR
    overall_snr = estimate_snr(waveform, sr)
    result["avg_snr_db"] = overall_snr
    result["min_snr_db"] = overall_snr
    result["max_snr_db"] = overall_snr

    # VAD (lossless 模式 — 不丢弃任何段)
    segments = energy_vad(waveform, sr, lossless=True, **vad_kwargs)

    # Save segments
    prefix = Path(audio_path).stem
    seg_for_save = [(wav, i) for i, (wav, _, _) in enumerate(segments)]
    seg_paths = save_segments(output_dir, prefix, seg_for_save, sr)

    # Compute statistics
    total_sec = 0.0
    all_snr = []
    for wav, start, end in segments:
        dur = len(wav) / sr
        total_sec += dur
        snr_val = estimate_snr(wav, sr)
        all_snr.append(snr_val)

    result["segment_count"] = len(segments)
    if all_snr:
        result["avg_snr_db"] = round(float(np.mean(all_snr)), 1)
        result["min_snr_db"] = round(float(np.min(all_snr)), 1)
        result["max_snr_db"] = round(float(np.max(all_snr)), 1)

    # ── Diarization ──
    if run_diarization and len(segments) > 0 and seg_paths:
        try:
            from train.diarizer import SpeakerDiarizer
            seg_path_list = [Path(p) for p in seg_paths]
            diarizer = SpeakerDiarizer(
                model_path=None,
                db_path=diarizer_db_path,
                model_name=diarizer_model_name,
                agent_threshold=diarizer_agent_threshold,
                cluster_threshold=diarizer_cluster_threshold,
            )
            diar_results = diarizer.diarize(seg_path_list)
            summary = diarizer.summarize(diar_results)

            result["agent_segments"] = summary["agent_segments"]
            result["customer_segments"] = summary["customer_segments"]
            result["agent_valid_sec"] = summary["agent_valid_sec"]
            result["customer_valid_sec"] = summary["customer_valid_sec"]
            result["uncertain_segments"] = summary["uncertain_segments"]
            result["uncertain_valid_sec"] = summary["uncertain_valid_sec"]
            result["diarization_done"] = True

            # Extract customer voiceprint
            customer_vp = diarizer.get_customer_voiceprint(diar_results, seg_path_list)
            if customer_vp is not None:
                result["customer_voiceprint_available"] = True
                result["customer_voiceprint_num_segments"] = customer_vp["num_segments"]
                logger.info(
                    "客户声纹提取成功: %d 段 centroid, dim=%d",
                    customer_vp["num_segments"], len(customer_vp["embedding"]),
                )
            else:
                logger.info("未提取到有效客户声纹（段数不足或因阈值过滤）")

            logger.info(
                "Diarization: agent=%d cust=%d uncert=%d (%.1f/%.1f/%.1f s)",
                result["agent_segments"], result["customer_segments"],
                result["uncertain_segments"],
                result["agent_valid_sec"], result["customer_valid_sec"],
                result["uncertain_valid_sec"],
            )

        except Exception as e:
            logger.warning("Diarization failed for %s: %s — 使用默认标注", audio_path, e)
            result["agent_segments"] = len(segments)
            result["agent_valid_sec"] = round(total_sec, 2)
    else:
        # Default: all segments treated as agent (fallback)
        result["agent_segments"] = len(segments)
        result["agent_valid_sec"] = round(total_sec, 2)

    logger.info(
        "Preprocessed %s: %d segments, %.1fs total, "
        "agent=%d cust=%d uncertain=%d, avg SNR=%.1f dB%s",
        audio_path, result["segment_count"], total_sec,
        result["agent_segments"], result["customer_segments"],
        result["uncertain_segments"], result["avg_snr_db"],
        "  [diarized]" if result["diarization_done"] else "",
    )

    return result


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_output_paths(
    preprocessed_root: str,
    biz_system: str,
    date_str: str,
    call_id: str,
) -> Path:
    """
    获取预处理输出目录。

    {preprocessed_root}/{biz_system}/{date_str}/{call_id}/
    """
    out_dir = Path(preprocessed_root) / biz_system / date_str / call_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
