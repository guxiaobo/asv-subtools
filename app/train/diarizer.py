"""
说话人分离模块 (Speaker Diarizer) —— 优化版。

在 VAD 切段后，对每个段做声纹鉴定：
1. 与数据库已存坐席声纹比较 → 标注坐席/客户候选
2. 客户候选段之间计算相似度 → 聚类确认客户声纹
3. 多说话人检测：当出现多个相似规模的群组时，取段数最大的群组为主客户
4. 客户声纹代表：使用 centroid 平均 embedding（非单一段）

设计要求（2026-06-08 修订）：
  - 所有 VAD 分段都不能丢失（lossless VAD）
  - 坐席声纹比对：高相似→坐席，低相似→客户候选
  - 客户段内聚类：至少 2 段以上高相似才确认为客户
  - 孤立段（与任何段都不相似）→ 暂标 uncertain
  - 多说话人场景：多个相似规模的群组→取段数最大的群组为客户
"""

from __future__ import annotations

import json
import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from train.config import load_config
from train.db import get_connection

logger = logging.getLogger("train.diarizer")

# 默认 ONNX 模型路径（相对于项目根）
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "api" / "models"

# ── Audio feature extraction (reused from verifier) ──


def load_wav_norm(path: Path) -> np.ndarray:
    """Load WAV as mono 16kHz float32, via soundfile (fallback ffmpeg).

    跨平台兼容：自动搜索 ffmpeg 可执行文件（PATH 环境变量）。
    """
    import shutil
    import soundfile as sf
    try:
        data, sr = sf.read(str(path))
    except Exception:
        # Try ffmpeg fallback
        import subprocess, io
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise FileNotFoundError(
                "ffmpeg not found in PATH; install ffmpeg or use a "
                "directly supported audio format"
            )
        r = subprocess.run(
            [ffmpeg_path, "-y", "-i", str(path),
             "-ar", "16000", "-ac", "1", "-f", "wav", "-loglevel", "error", "pipe:1"],
            capture_output=True, timeout=30
        )
        data, sr = sf.read(io.BytesIO(r.stdout))
    if sr != 16000:
        import subprocess, io
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise FileNotFoundError(
                "ffmpeg not found in PATH; install ffmpeg or use a "
                "directly supported audio format"
            )
        r = subprocess.run(
            [ffmpeg_path, "-y", "-i", str(path),
             "-ar", "16000", "-ac", "1", "-f", "wav", "-loglevel", "error", "pipe:1"],
            capture_output=True, timeout=30
        )
        data, sr = sf.read(io.BytesIO(r.stdout))
    return data.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """Create mel filterbank matrix (n_mels × n_fft//2+1) without librosa."""
    low_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m]):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(bins[m], bins[m + 1]):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank


def _stft_power(audio: np.ndarray, n_fft: int = 400,
                hop_length: int = 160) -> np.ndarray:
    """Compute power spectrogram without librosa. Returns (n_fft//2+1, T)."""
    window = np.hamming(n_fft)
    n_frames = max(1, 1 + (len(audio) - n_fft) // hop_length)
    result = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for t in range(n_frames):
        start = t * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.fft.rfft(frame * window)
        result[:, t] = np.abs(spec) ** 2
    return result


def extract_fbank(audio: np.ndarray, sr: int = 16000,
                  num_filters: int = 80) -> np.ndarray:
    """Extract log-mel filterbank features, shape (T, 80). No librosa."""
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    power = _stft_power(audio, n_fft=400, hop_length=160)
    mel_basis = _mel_filterbank(sr, 400, num_filters, 20.0, 7600.0)
    mel = mel_basis @ power
    return np.log(np.maximum(mel, 1e-10)).T


# ── Helper: connected components via BFS ──


def _connected_components(
    adj: np.ndarray,
    min_size: int = 1,
) -> List[List[int]]:
    """
    从邻接矩阵中找到连通分量，按大小降序排列。

    Args:
        adj: m×m 布尔邻接矩阵（不含自环）。
        min_size: 最小分量大小，< 此值忽略。

    Returns:
        分量列表，按大小降序排列，每个分量是节点索引列表。
    """
    m = adj.shape[0]
    visited = set()
    components = []

    for start in range(m):
        if start in visited:
            continue
        # BFS
        queue = [start]
        comp = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for neighbor in range(m):
                if adj[node, neighbor] and neighbor not in visited:
                    queue.append(neighbor)
        if len(comp) >= min_size:
            components.append(comp)

    components.sort(key=lambda c: len(c), reverse=True)
    return components


def _pairwise_similarity(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    计算一批 L2 归一化 embedding 的成对余弦相似度矩阵。

    Returns:
        m×m 相似度矩阵（对角线为 1.0）。
    """
    m = len(embeddings)
    mat = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            s = float(np.dot(embeddings[i], embeddings[j]))
            mat[i, j] = s
            mat[j, i] = s
    return mat


def _centroid_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    计算质心 embedding：对多个 L2 归一化向量取平均后重新归一化。

    这是最稳健的说话人声纹表示方式：
    - 使用所有段的信息，比单个段更鲁棒
    - 平均操作降低噪声影响
    - 重归一化保证输出仍是单位向量
    """
    if not embeddings:
        raise ValueError("embeddings 列表不能为空")
    if len(embeddings) == 1:
        return embeddings[0].copy()
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


# ── Speaker Diarizer (优化版) ──


class SpeakerDiarizer:
    """VAD 段标注器：比对坐席声纹 + 客户段内聚类（优化版）。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: Optional[str] = None,
        model_name: str = "CAM++",
        agent_threshold: Optional[float] = None,
        cluster_threshold: float = 0.55,
        min_samples: int = 2,
    ):
        """
        Args:
            model_path: ONNX 模型路径（默认 api/models/campplus.onnx）。
            db_path: SQLite 路径（默认自动查找）。
            model_name: 从 DB 加载哪个模型的坐席声纹。
            agent_threshold: 坐席判定阈值（默认 None=动态检测）。
            cluster_threshold: 客户聚类相似度阈值。
            min_samples: 聚类最少段数才确认为客户。
        """
        # Resolve model path based on model_name (if not explicitly set)
        if not model_path:
            model_map = {
                "CAM++": _DEFAULT_MODEL_DIR / "campplus.onnx",
                "ResNet34": _DEFAULT_MODEL_DIR / "voxceleb_resnet34_LM.onnx",
                "ECAPA": _DEFAULT_MODEL_DIR / "ecapa-speaker-v1.onnx",
            }
            model_path_obj = model_map.get(model_name, _DEFAULT_MODEL_DIR / "campplus.onnx")
        else:
            model_path_obj = Path(model_path)
        self.model_path = model_path_obj
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.model_path}")

        # Load ONNX
        warnings.filterwarnings("ignore", category=UserWarning)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self._has_feature_lens = len(self.session.get_inputs()) > 1

        # Load agent reference from DB
        self.db_path = db_path
        self.model_name = model_name
        self.agent_ref = self._load_agent_ref(db_path, model_name)
        self.agent_dim = len(self.agent_ref) if self.agent_ref is not None else 0

        # Threshold
        if agent_threshold is not None:
            self.agent_threshold = agent_threshold
        else:
            # Per-model default thresholds (calibrated via experiments)
            model_name_lower = self.model_path.name.lower()
            if "campplus" in model_name_lower or "cam++" in model_name_lower:
                self.agent_threshold = 0.49
            elif "resnet34" in model_name_lower:
                self.agent_threshold = 0.59
            elif "ecapa" in model_name_lower:
                self.agent_threshold = 0.68
            else:
                self.agent_threshold = 0.55  # generic default
        # Per-model cluster threshold defaults (model-specific calibration)
        # CAM++ on telephone audio: customer inter-segment similarity ~0.25-0.50,
        # so lower threshold needed vs ResNet34/ECAPA which tend to overshoot.
        if cluster_threshold == 0.55:  # still at default → apply per-model
            model_name_lower = self.model_path.name.lower()
            if "campplus" in model_name_lower or "cam++" in model_name_lower:
                self.cluster_threshold = 0.35
            elif "ecapa" in model_name_lower:
                self.cluster_threshold = 0.55
            elif "resnet34" in model_name_lower:
                self.cluster_threshold = 0.55
            else:
                self.cluster_threshold = cluster_threshold
        else:
            self.cluster_threshold = cluster_threshold
        self.min_samples = min_samples
        # Adaptive threshold tuning: if True, adjusts agent_threshold based on
        # the score distribution of the current call (only when segments < 200)
        self.adaptive_tuning = True

        logger.info(
            "Diarizer ready: model=%s agent_dim=%d threshold=%.2f cluster=%.2f",
            self.model_path.name, self.agent_dim,
            self.agent_threshold, self.cluster_threshold,
        )

    # ── Agent reference loading ──

    def _load_agent_ref(self, db_path: Optional[str],
                        model_name: str) -> Optional[np.ndarray]:
        """从 speaker_voiceprints 表加载坐席声纹。"""
        if not db_path:
            cfg = load_config()
            db_path = cfg.get("db_path")
        conn = get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT embedding FROM speaker_voiceprints "
                "WHERE model_name=? AND speaker_type='agent' "
                "ORDER BY id DESC LIMIT 1",
                (model_name,)
            ).fetchone()
            if not row:
                logger.warning("DB 中未找到坐席声纹 (model_name=%s)", model_name)
                return None
            emb = np.frombuffer(row[0], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            logger.info("已加载坐席声纹: dim=%d", len(emb))
            return emb
        finally:
            conn.close()

    # ── Embedding extraction ──

    def extract_embedding(self, wav_path: Path) -> Optional[np.ndarray]:
        """从 WAV 文件提取 L2 归一化 embedding。"""
        try:
            audio = load_wav_norm(wav_path)
            if len(audio) < 160:  # <10ms → 太短
                logger.debug("段太短: %s (%d samples)", wav_path.name, len(audio))
                return None
            fbank = extract_fbank(audio)
            if fbank.shape[0] < 5:
                logger.debug("fbank 帧太少: %s (%d frames)", wav_path.name, fbank.shape[0])
                return None

            inp = fbank[np.newaxis, ...].astype(np.float32)
            if self._has_feature_lens:
                lens = np.array([fbank.shape[0]], dtype=np.float32)
                out = self.session.run(
                    None, {self.input_name: inp, "feature_lens": lens}
                )[0]
            else:
                out = self.session.run(None, {self.input_name: inp})[0]

            emb = out.flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.warning("Embedding 提取失败: %s → %s", wav_path.name, e)
            return None

    # ── Per-segment diarization (优化版) ──

    def _auto_detect_threshold(self, scores: List[float]) -> float:
        """
        Analyze the score distribution of all segments in the current call
        and find the best threshold to separate agent from customer.

        策略：寻找得分分布中最大间隔点（Score 排序后相邻差值最大处）。
        这对于电话录音中的两端（坐席/客户）对话特别有效。

        Args:
            scores: 所有段与坐席参考的余弦相似度列表（不含 None）

        Returns:
            自适应阈值（介于 0.3–0.9 之间）
        """
        if len(scores) < 5:
            return self.agent_threshold  # 段太少，使用默认值
        scores_np = np.sort(scores)
        gaps = np.diff(scores_np)
        # 平滑窗口（取最大间隔附近的 3 个平均）
        window = min(3, len(gaps))
        kernel = np.ones(window) / window
        smooth_gaps = np.convolve(gaps, kernel, mode="same")
        max_gap_idx = int(np.argmax(smooth_gaps))
        threshold = float(scores_np[max_gap_idx])
        # 夹在合理范围
        return float(np.clip(threshold, 0.30, 0.90))

    def _otsu_threshold(self, scores: List[float]) -> Optional[float]:
        """
        Otsu 双峰分割：找让组内加权方差最小的分割点。

        电话录音 sim_to_agent 通常呈双峰分布：
          - 高峰（坐席段，sim 通常 0.75+）
          - 低峰（客户段，sim 通常 0.35-0.70）
        Otsu 自动找两峰之间的最佳分割阈值。
        """
        if len(scores) < 4:
            return None

        arr = sorted(scores)
        best_t = None
        best_var = float("inf")

        for i in range(1, len(arr)):
            t = (arr[i - 1] + arr[i]) / 2.0
            g1 = [s for s in arr if s <= t]
            g2 = [s for s in arr if s > t]
            if len(g1) < 2 or len(g2) < 2:
                continue
            w1 = len(g1) / len(arr)
            w2 = len(g2) / len(arr)
            var = w1 * np.var(g1) + w2 * np.var(g2)
            if var < best_var:
                best_var = var
                best_t = t

        if best_t is not None:
            best_t = float(np.clip(best_t, 0.30, 0.90))
        return best_t

    def diarize(self, segment_files: List[Path]) -> List[Dict[str, Any]]:
        """
        对一批 VAD 段 WAV 文件执行说话人标注（v4: Otsu 双峰分割）。

        策略演进:
          v1: 固定阈值 sim_to_agent → 阈值切分
          v2: 自适应阈值（最大间隔法）
          v3: pairwise 聚类优先（失败 — 客户段 pairwise 太分散）
          v4: Otsu 双峰分割（利用 sim_to_agent 双峰分布特性）

        核心发现:
          电话录音中，客户段与坐席声纹的 cosine 也很高（0.5-0.7），
          所以不能用绝对阈值。但 sim_to_agent 分布呈双峰（坐席高峰
          vs 客户低峰），Otsu 法能自动找到最佳分割点。

        算法流程:
          1. 提取所有段 embedding
          2. 计算 sim_to_agent
          3. Otsu 找分割阈值 → 坐席/客户二分
          4. 客户段内 pairwise 聚类确认（去掉误标段）
          5. 客户声纹用 centroid 平均 embedding

        Returns:
            [{
                "idx": int,
                "file": str,
                "label": "agent" | "customer" | "uncertain",
                "sim_to_agent": float | None,
                "customer_cluster_id": int | None,
                "customer_centroid_used": bool,
                "nearest_to_centroid": bool,
                "dur": float,
            }, ...]
        """
        if not segment_files:
            return []

        n = len(segment_files)

        # ── Step 1: 提取所有 embedding + 时长 ──
        embs: List[Optional[np.ndarray]] = []
        durs: List[float] = []
        for f in segment_files:
            emb = self.extract_embedding(f)
            dur = 0.0
            if emb is not None:
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(f))
                    dur = len(data) / sr
                except Exception:
                    pass
            else:
                logger.debug("段无法提取 embedding: %s", f.name)
            embs.append(emb)
            durs.append(round(dur, 2))

        # ── Step 2: 计算 sim_to_agent ──
        all_sims: List[Optional[float]] = []
        for i, emb in enumerate(embs):
            if emb is not None and self.agent_ref is not None:
                all_sims.append(round(float(np.dot(emb, self.agent_ref)), 4))
            else:
                all_sims.append(None)

        # 初始化 results（默认全部 agent）
        results = []
        for i in range(n):
            results.append({
                "idx": i,
                "file": segment_files[i].name,
                "label": "agent",
                "sim_to_agent": all_sims[i],
                "customer_cluster_id": None,
                "customer_centroid_used": False,
                "nearest_to_centroid": False,
                "dur": durs[i],
            })

        # 有效 embedding 的索引
        valid_idx = [i for i in range(n) if embs[i] is not None]
        if len(valid_idx) < 3:
            # 段太少，无法双峰分割
            return results

        # ── Step 3: Otsu 双峰分割 → 坐席/客户二分 ──
        valid_sims = [all_sims[i] for i in valid_idx]
        # 过滤 None（不应该有，但防御）
        valid_sims_clean = [s for s in valid_sims if s is not None]

        otsu_t = self._otsu_threshold(valid_sims_clean) if len(valid_sims_clean) >= 4 else None

        if otsu_t is None:
            # 不足以做双峰分割，fallback 到固定阈值
            otsu_t = self.agent_threshold
            logger.debug("Otsu 不可用 (%d 有效段)，fallback 阈值=%.2f", len(valid_sims_clean), otsu_t)

        logger.info(
            "Otsu 分割阈值=%.3f (基于 %d 段, sim 范围 %.3f~%.3f)",
            otsu_t, len(valid_sims_clean),
            min(valid_sims_clean), max(valid_sims_clean),
        )

        # 按 Otsu 阈值分割：低 sim → 客户候选, 高 sim → 坐席
        cust_candidate_indices = []  # valid_idx 中的下标
        for vi, ri in enumerate(valid_idx):
            s = all_sims[ri]
            if s is not None and s < otsu_t:
                results[ri]["label"] = "customer_candidate"
                cust_candidate_indices.append(vi)

        if not cust_candidate_indices:
            # 没有低于阈值的段 — 整通都是坐席（可能是纯坐席录音）
            logger.info("Otsu 阈值=%.3f 以下无段，整通标注为坐席", otsu_t)
            return results

        logger.info(
            "Otsu 分割: 阈值=%.3f, 客户候选=%d 段, 坐席=%d 段",
            otsu_t, len(cust_candidate_indices), len(valid_idx) - len(cust_candidate_indices),
        )

        # ── Step 4: 客户候选段内部 pairwise 聚类 → 确认 ──
        # 过滤有效 embedding 的客户候选
        valid_cust = [
            (vi, embs[valid_idx[vi]])
            for vi in cust_candidate_indices
            if embs[valid_idx[vi]] is not None
        ]

        if len(valid_cust) < self.min_samples:
            # 客户候选太少，无法聚类确认
            # 但 Otsu 分割本身就可靠，直接标注为 customer
            for vi, _ in valid_cust:
                ri = valid_idx[vi]
                results[ri]["label"] = "customer"
                results[ri]["customer_cluster_id"] = 0
                results[ri]["customer_centroid_used"] = True
            return results

        # pairwise 聚类
        cust_vectors = [e for _, e in valid_cust]
        cust_vi_map = [vi for vi, _ in valid_cust]  # valid_idx 中的位置
        sim_mat = _pairwise_similarity(cust_vectors)

        # 用 cluster_threshold 做连通分量
        adj = sim_mat >= self.cluster_threshold
        np.fill_diagonal(adj, False)
        components = _connected_components(adj, min_size=self.min_samples)

        if components:
            # 有满足 min_samples 的群组 → 取最大的作为主客户
            main_comp = components[0]
            comp_vecs = [cust_vectors[i] for i in main_comp]
            centroid = _centroid_embedding(comp_vecs)

            for mi in main_comp:
                ri = valid_idx[cust_vi_map[mi]]
                results[ri]["label"] = "customer"
                results[ri]["customer_cluster_id"] = 0
                results[ri]["customer_centroid_used"] = True

            # 标记最近 centroid 的段
            best_dist = -1
            best_mi = main_comp[0]
            for mi in main_comp:
                d = float(np.dot(cust_vectors[mi], centroid))
                if d > best_dist:
                    best_dist = d
                    best_mi = mi
            results[valid_idx[cust_vi_map[best_mi]]]["nearest_to_centroid"] = True

            # 不在主群组中的客户候选 → 回退为 uncertain
            in_main = set(main_comp)
            for ci in range(len(valid_cust)):
                if ci not in in_main:
                    ri = valid_idx[cust_vi_map[ci]]
                    results[ri]["label"] = "uncertain"

            # 多说话人：其他足够大的群组 → uncertain (不同客户)
            for gi, comp in enumerate(components[1:], 1):
                for mi in comp:
                    ri = valid_idx[cust_vi_map[mi]]
                    results[ri]["label"] = "uncertain"
                    results[ri]["customer_cluster_id"] = gi
        else:
            # 没有 min_samples 的群组 — 但 Otsu 分割了这些段
            # 在跨录音聚合时这些段仍然有用，直接标 customer
            for vi, _ in valid_cust:
                ri = valid_idx[vi]
                results[ri]["label"] = "customer"
                results[ri]["customer_cluster_id"] = 0
                results[ri]["customer_centroid_used"] = True

        # 无效 embedding → uncertain
        for i in range(n):
            if embs[i] is None:
                results[i]["label"] = "uncertain"

        return results

    # ── Legacy diarize (sim_to_agent threshold) ──

    def _diarize_legacy(self, segment_files: List[Path]) -> List[Dict[str, Any]]:
        """旧版 diarize: sim_to_agent 阈值切分（保留作为 fallback）。"""
        if not segment_files:
            return []

        n = len(segment_files)
        embs = []
        for f in segment_files:
            emb = self.extract_embedding(f)
            dur = 0.0
            if emb is not None:
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(f))
                    dur = len(data) / sr
                except Exception:
                    pass
            embs.append((emb, dur))

        all_sims = []
        for i, (emb, dur) in enumerate(embs):
            if emb is not None and self.agent_ref is not None:
                sim = float(np.dot(emb, self.agent_ref))
                all_sims.append((i, sim))
            else:
                all_sims.append((i, None))

        actual_scores = [s[1] for s in all_sims if s[1] is not None]
        if self.adaptive_tuning and len(actual_scores) >= 5:
            detected = self._auto_detect_threshold(actual_scores)
            if abs(detected - self.agent_threshold) > 0.02:
                self.agent_threshold = detected

        results = []
        cust_candidate_indices = []

        for i, (emb, dur) in enumerate(embs):
            entry = {
                "idx": i,
                "file": segment_files[i].name,
                "label": "uncertain",
                "sim_to_agent": None,
                "customer_cluster_id": None,
                "customer_centroid_used": False,
                "nearest_to_centroid": False,
                "dur": round(dur, 2),
            }
            sim = all_sims[i][1]
            if sim is not None:
                entry["sim_to_agent"] = round(sim, 4)
                if sim >= self.agent_threshold:
                    entry["label"] = "agent"
                else:
                    entry["label"] = "customer_candidate"
                    cust_candidate_indices.append(i)
            elif emb is not None:
                entry["label"] = "customer_candidate"
                cust_candidate_indices.append(i)
            results.append(entry)

        if len(cust_candidate_indices) >= 2:
            # 过滤出有效 embedding 的客户候选
            valid_pairs = [
                (i, embs[i][0]) for i in cust_candidate_indices
                if embs[i][0] is not None
            ]
            if len(valid_pairs) >= 2:
                indices_arr = [p[0] for p in valid_pairs]  # actual results index
                vectors = [p[1] for p in valid_pairs]      # embedding vectors

                # 3a. 成对相似度矩阵
                sim_mat = _pairwise_similarity(vectors)

                # 3b. 邻接图（阈值过滤，去掉自环）
                adj = sim_mat >= self.cluster_threshold
                np.fill_diagonal(adj, False)

                # 3c. 连通分量聚类（按大小降序排列）
                components = _connected_components(adj, min_size=self.min_samples)

                if len(components) == 0:
                    # 所有候选段都无法形成有效聚类（互相都不够相似）
                    # 全部标 uncertain，不做客户声纹提取
                    logger.info(
                        "客户候选段未形成有效聚类: %d 段互相最高相似度 %.4f "
                        "(阈值 %.2f)",
                        len(vectors),
                        float(np.max(sim_mat)) if sim_mat.size > 0 else 0,
                        self.cluster_threshold,
                    )
                else:
                    # 取最大群组为主客户
                    main_comp = components[0]
                    main_size = len(main_comp)
                    total_candidate = len(valid_pairs)

                    # 检测多说话人场景：
                    #   - 第二大群组 ≥ 主群组大小的 60%
                    #   - 或存在 3 个以上的群组
                    #   - 或未聚类段（孤立段）特别多（≥主群组 50% 且主群组不多）
                    other_sizes = [len(c) for c in components[1:]]
                    num_components = len(components)
                    isolated_count = total_candidate - sum(len(c) for c in components)
                    has_multi_speaker = False
                    multi_speaker_reason = ""

                    if other_sizes:
                        max_other = max(other_sizes)
                        if max_other >= main_size * 0.6:
                            has_multi_speaker = True
                            multi_speaker_reason = (
                                f"第二群组{max_other}/{main_size} >= 60%"
                            )
                    if not has_multi_speaker and num_components >= 3 and main_size <= 3:
                        has_multi_speaker = True
                        multi_speaker_reason = (
                            f"主群组仅{main_size}段, 却有{num_components}个群组"
                        )
                    if not has_multi_speaker and isolated_count >= main_size * 0.5 and main_size <= 3:
                        has_multi_speaker = True
                        multi_speaker_reason = (
                            f"孤立段{isolated_count} ≥ 主群组{main_size}的50%"
                        )

                    if has_multi_speaker:
                        logger.info(
                            "多说话人检测: 主群组=%d 段, 其他=%s, 孤立=%d, 原因=%s",
                            main_size, other_sizes, isolated_count, multi_speaker_reason,
                        )

                    # ── 计算主客户 centroid embedding ──
                    main_vectors = [vectors[i] for i in main_comp]
                    customer_centroid = _centroid_embedding(main_vectors)

                    # 如果需要选单一段（配合旧系统或其他原因），选与 centroid 最近的段
                    main_sims_to_centroid = [
                        (i, float(np.dot(vectors[i], customer_centroid)))
                        for i in main_comp
                    ]
                    best_single_idx = max(
                        main_sims_to_centroid, key=lambda x: x[1]
                    )[0]

                    # 标记主群组为客户
                    cluster_id = 0
                    for i in main_comp:
                        idx_in_results = indices_arr[i]
                        results[idx_in_results]["label"] = "customer"
                        results[idx_in_results]["customer_cluster_id"] = cluster_id
                        results[idx_in_results]["customer_centroid_used"] = True
                        # 记录最近 centroid 的段（供后续参考）
                        results[idx_in_results]["nearest_to_centroid"] = (
                            i == best_single_idx
                        )

                    # 多说话人场景：非主群组 → uncertain
                    if has_multi_speaker:
                        for comp in components[1:]:
                            cluster_id += 1
                            for i in comp:
                                idx_in_results = indices_arr[i]
                                results[idx_in_results]["label"] = "uncertain"
                                results[idx_in_results]["customer_cluster_id"] = None
                                results[idx_in_results]["customer_centroid_used"] = False
                                results[idx_in_results]["nearest_to_centroid"] = False

                    # 孤立段（不属于任何聚类的候选段）→ uncertain
                    all_clustered = set()
                    for comp in components:
                        for i in comp:
                            all_clustered.add(i)
                    for node in range(len(vectors)):
                        if node not in all_clustered:
                            idx_in_results = indices_arr[node]
                            results[idx_in_results]["label"] = "uncertain"
                            results[idx_in_results]["nearest_to_centroid"] = False

        elif len(cust_candidate_indices) == 1:
            # 只有 1 个客户候选段 → 必然无法形成聚类，标 uncertain
            idx = cust_candidate_indices[0]
            results[idx]["label"] = "uncertain"
            logger.debug("仅有 1 个客户候选段，无法聚类确认（要求至少2段以上互相验证）")

        # Step 4: Final cleanup — rename remaining customer_candidate -> uncertain
        for r in results:
            if r["label"] == "customer_candidate":
                r["label"] = "uncertain"

        return results

    # ── 客户声纹提取 ──

    def get_customer_voiceprint(
        self, diarize_results: List[Dict[str, Any]], segment_files: List[Path]
    ) -> Optional[Dict[str, Any]]:
        """
        从 diarize 结果中提取客户声纹（centroid 平均 embedding）。

        只有当存在被标记为 customer 且属于主群组的段时，才返回有效声纹。

        Returns:
            {
                "embedding": np.ndarray,  # L2 归一化 centroid
                "num_segments": int,       # 贡献段数
                "cluster_id": int,         # 群组 ID
                "segments": List[str],     # 贡献段文件名列表
            }
            如果无有效客户段，返回 None。
        """
        # 找出被标记为 customer 且是 centroid 贡献的段
        customer_segs = [
            r for r in diarize_results
            if r["label"] == "customer" and r["customer_centroid_used"]
        ]
        if not customer_segs:
            return None

        # 取这些段的 embedding（重新提取，因为 diarize 方法不存 embedding）
        embs = []
        seg_names = []
        for r in customer_segs:
            wav_path = segment_files[r["idx"]]
            emb = self.extract_embedding(wav_path)
            if emb is not None:
                embs.append(emb)
                seg_names.append(r["file"])

        if len(embs) < self.min_samples:
            return None

        centroid = _centroid_embedding(embs)

        return {
            "embedding": centroid,
            "num_segments": len(embs),
            "cluster_id": customer_segs[0].get("customer_cluster_id", 0),
            "segments": seg_names,
        }

    # ── Cross-call customer aggregation ──

    def cross_call_aggregate(
        self,
        calls_by_customer: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        跨同客户多通录音聚合客户声纹。

        对每个客户的多通录音，汇总所有非坐席段（customer + uncertain）的
        embedding，跨录音聚类，满足 min_samples 的群组确认为该客户的声纹。

        Args:
            calls_by_customer: {客户ID: [call_info, ...]}
                每个 call_info 是一个字典：
                {
                    "call_id": str,           # 录音 ID（文件名）
                    "segment_files": [Path],  # 该通录音的所有 VAD 段文件
                    "diarize_results": [...], # 该通录音的逐段标注结果
                }

        Returns:
            [{
                "customer_id": str,
                "embedding": np.ndarray,      # L2 归一化 centroid
                "num_segments": int,
                "num_calls": int,             # 贡献的录音数
                "source_call_ids": [str],
                "segments": [{"call_id": str, "file": str}, ...],
            }, ...]
        """
        results = []

        for customer_id, calls in calls_by_customer.items():
            # 收集该客户所有通话中的非坐席段
            non_agent_entries = []  # [(embedding, call_id, file_name)]

            for call_info in calls:
                call_id = call_info["call_id"]
                seg_files = call_info["segment_files"]
                diar_results = call_info["diarize_results"]

                for r in diar_results:
                    if r["label"] in ("customer", "uncertain", "customer_candidate"):
                        seg_path = seg_files[r["idx"]]
                        emb = self.extract_embedding(seg_path)
                        if emb is not None:
                            non_agent_entries.append((emb, call_id, r["file"]))

            if len(non_agent_entries) < self.min_samples:
                logger.info(
                    "跨录音聚合 [%s]: 仅 %d 个非坐席段（需≥%d），跳过",
                    customer_id, len(non_agent_entries), self.min_samples,
                )
                continue

            # 跨录音 pairwise similarity + 聚类
            vectors = [e[0] for e in non_agent_entries]
            sim_mat = _pairwise_similarity(vectors)
            adj = sim_mat >= self.cluster_threshold
            np.fill_diagonal(adj, False)

            components = _connected_components(adj, min_size=self.min_samples)

            if not components:
                # 所有段互相不够相似
                max_sim = float(np.max(sim_mat)) if sim_mat.size > 0 else 0
                logger.info(
                    "跨录音聚合 [%s]: %d 段互相最高相似度 %.4f（阈值 %.2f），无法聚类",
                    customer_id, len(vectors), max_sim, self.cluster_threshold,
                )
                continue

            # 取最大群组为客户声纹
            main_comp = components[0]
            main_vectors = [vectors[i] for i in main_comp]
            centroid = _centroid_embedding(main_vectors)

            # 汇总来源信息
            contributing_calls = set()
            seg_details = []
            for i in main_comp:
                _, call_id, file_name = non_agent_entries[i]
                contributing_calls.add(call_id)
                seg_details.append({"call_id": call_id, "file": file_name})

            result_entry = {
                "customer_id": customer_id,
                "embedding": centroid,
                "num_segments": len(main_comp),
                "num_calls": len(contributing_calls),
                "source_call_ids": sorted(contributing_calls),
                "segments": seg_details,
            }
            results.append(result_entry)

            logger.info(
                "跨录音聚合 [%s]: ✅ %d 段 / %d 通录音 → centroid (dim=%d)",
                customer_id, len(main_comp), len(contributing_calls),
                len(centroid),
            )

            # 如果有多个群组（多说话人场景），记录但只取最大群组
            if len(components) > 1:
                other_sizes = [len(c) for c in components[1:]]
                logger.info(
                    "跨录音聚合 [%s]: 检测到 %d 个群组，最大=%d段，其他=%s",
                    customer_id, len(components), len(main_comp), other_sizes,
                )

        return results

    # ── Statistics ──

    def summarize(self, results: List[Dict]) -> Dict[str, Any]:
        """从 diarize 结果汇总统计。"""
        agent_segs = [r for r in results if r["label"] == "agent"]
        customer_segs = [r for r in results if r["label"] == "customer"]
        uncertain_segs = [r for r in results if r["label"] == "uncertain"]

        clusters = set(
            r["customer_cluster_id"] for r in customer_segs
            if r["customer_cluster_id"] is not None
        )

        return {
            "segment_count": len(results),
            "agent_segments": len(agent_segs),
            "customer_segments": len(customer_segs),
            "customer_clusters": len(clusters),
            "uncertain_segments": len(uncertain_segs),
            "agent_valid_sec": round(sum(r["dur"] for r in agent_segs), 2),
            "customer_valid_sec": round(sum(r["dur"] for r in customer_segs), 2),
            "uncertain_valid_sec": round(sum(r["dur"] for r in uncertain_segs), 2),
            "avg_sim_to_agent_agent": round(
                np.mean([r["sim_to_agent"] for r in agent_segs if r["sim_to_agent"] is not None]), 4
            ) if agent_segs else 0,
            "avg_sim_to_agent_customer": round(
                np.mean([r["sim_to_agent"] for r in customer_segs if r["sim_to_agent"] is not None]), 4
            ) if customer_segs else 0,
        }


# ── Standalone usage ──


def main():
    """CLI: 对已有预处理段重新跑说话人标注。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="对已有 VAD 段执行说话人标注（优化版）"
    )
    parser.add_argument("--model-path", default=None, help="ONNX 模型路径")
    parser.add_argument("--db-path", default=None, help="SQLite 路径")
    parser.add_argument("--model-name", default="CAM++", help="DB 中声纹的 model_name")
    parser.add_argument("--threshold", type=float, default=None, help="坐席判定阈值（默认 None=自动检测）")
    parser.add_argument("--cluster-threshold", type=float, default=0.55, help="客户聚类阈值")
    parser.add_argument("--target-dir", default=None, help="指定扫描的 segments 目录")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    diarizer = SpeakerDiarizer(
        model_path=args.model_path,
        db_path=args.db_path,
        model_name=args.model_name,
        agent_threshold=args.threshold,
        cluster_threshold=args.cluster_threshold,
    )

    # Find segment WAV files
    if args.target_dir:
        seg_files = sorted(Path(args.target_dir).rglob("*_seg*.wav"))
    else:
        # Default: scan data/preprocessed
        root = Path(__file__).resolve().parent.parent / "data" / "preprocessed"
        seg_files = sorted(root.rglob("*_seg*.wav"))

    logger.info("共找到 %d 个段文件", len(seg_files))
    if not seg_files:
        return

    # Group by call (parent directory = call directory)
    from collections import defaultdict
    call_groups = defaultdict(list)
    for f in seg_files:
        call_dir = f.parent
        call_groups[str(call_dir)].append(f)

    print(f"\n{'='*60}")
    print(f"Diarizer 分析结果（优化版）")
    print(f"{'='*60}")

    total_agent = 0
    total_customer = 0
    total_uncertain = 0

    for call_dir, files in sorted(call_groups.items()):
        results = diarizer.diarize(files)
        summary = diarizer.summarize(results)
        call_name = Path(call_dir).name

        print(f"\n  {call_name} ({len(files)} 段):")
        print(f"    坐席={summary['agent_segments']} 客户={summary['customer_segments']} "
              f"不确定={summary['uncertain_segments']}")
        print(f"    坐席时长={summary['agent_valid_sec']:.1f}s "
              f"客户时长={summary['customer_valid_sec']:.1f}s")

        # 提取客户声纹
        customer_vp = diarizer.get_customer_voiceprint(results, files)
        if customer_vp:
            print(f"    → 客户声纹: {customer_vp['num_segments']} 段 centroid, "
                  f"dim={len(customer_vp['embedding'])}")
        else:
            print(f"    → 未提取到有效客户声纹")

        for r in results:
            if r["label"] == "agent":
                marker = "🟢"
            elif r["label"] == "customer":
                marker = "🔴"
            else:
                marker = "⚪"
            centroid_flag = " [C]" if r.get("customer_centroid_used") else ""
            print(f"      {marker} {r['file']}: {r['label']:8s}{centroid_flag} "
                  f"sim={r['sim_to_agent']:.4f} dur={r['dur']:.1f}s")

        total_agent += summary["agent_segments"]
        total_customer += summary["customer_segments"]
        total_uncertain += summary["uncertain_segments"]

    print(f"\n{'='*60}")
    print(f"合计: 坐席={total_agent} 客户={total_customer} 不确定={total_uncertain}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
