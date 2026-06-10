#!/usr/bin/env python3
"""
增量训练对比实验脚本 — v2 (自适应阈值).

对比三个 ONNX 模型（CAM++, ResNet34, ECAPA-TDNN）的
"旧版 enroll（DB 现有声纹）" vs "新版 enroll（从当前标注段计算的 centroid）" 的得分差异。

关键改进 v2：
- 自适应坐席判定阈值（从得分分布的间隙自动寻找最佳分隔点）
- 所有段不丢失（lossless）
- PLDA 训练跳过阈值不稳定场景
- 显示三个模型在同一套测试集上的可比结果

Usage:
    cd /Users/guxiaobo/Documents/GitHub/asv-subtools
    python3 -m app.train.incremental_compare
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("incremental_compare")

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # asv-subtools/
MODELS_DIR = PROJ_ROOT / "app" / "api" / "models"
DB_PATH = PROJ_ROOT / "app" / "data" / "training.db"
PREPROCESSED_ROOT = PROJ_ROOT / "app" / "data" / "preprocessed"

MODEL_CONFIGS = [
    {
        "name": "CAM++",
        "onnx_path": MODELS_DIR / "campplus.onnx",
        "embedding_dim": 192,
    },
    {
        "name": "ResNet34",
        "onnx_path": MODELS_DIR / "voxceleb_resnet34_LM.onnx",
        "embedding_dim": 256,
    },
    {
        "name": "ECAPA",
        "onnx_path": MODELS_DIR / "ecapa-speaker-v1.onnx",
        "embedding_dim": 192,
    },
]


# ─── Feature extraction (no librosa dependency) ────────────────────────────

def load_wav_norm(path: Path) -> np.ndarray:
    """Load WAV as mono 16kHz float32."""
    import soundfile as sf
    try:
        data, sr = sf.read(str(path))
    except Exception:
        import io, subprocess
        ffmpeg = "/opt/homebrew/bin/ffmpeg"
        r = subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1",
             "-f", "wav", "-loglevel", "error", "pipe:1"],
            capture_output=True, timeout=30,
        )
        data, sr = sf.read(io.BytesIO(r.stdout))
    if sr != 16000:
        import io, subprocess
        ffmpeg = "/opt/homebrew/bin/ffmpeg"
        r = subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1",
             "-f", "wav", "-loglevel", "error", "pipe:1"],
            capture_output=True, timeout=30,
        )
        data, sr = sf.read(io.BytesIO(r.stdout))
    return data.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
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
    """Returns (n_fft//2+1, T)."""
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
    """Extract log-mel filterbank features, shape (T, 80)."""
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    power = _stft_power(audio, n_fft=400, hop_length=160)
    mel_basis = _mel_filterbank(sr, 400, num_filters, 20.0, 7600.0)
    mel = mel_basis @ power
    return np.log(np.maximum(mel, 1e-10)).T


# ─── ONNX wrapper ──────────────────────────────────────────────────────────

class OnnxExtractor:
    """Extract L2-normalized embeddings from ONNX model."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._has_feature_lens = len(self.session.get_inputs()) > 1

    def extract(self, wav_path: Path) -> Optional[np.ndarray]:
        try:
            audio = load_wav_norm(wav_path)
            if len(audio) < 160:
                return None
            fbank = extract_fbank(audio)
            if fbank.shape[0] < 5:
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
            nrm = np.linalg.norm(emb)
            return emb / nrm if nrm > 0 else None
        except Exception as e:
            logger.debug("Extract fail: %s — %s", wav_path.name, e)
            return None


# ─── EER / minDCF ──────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    if len(scores) < 2:
        return 1.0, 0.5, 1.0
    idx = np.argsort(scores)[::-1]
    scores, labels = scores[idx], labels[idx]
    pos, neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return 1.0, 0.5, 1.0
    far = 1.0 - np.cumsum(labels == 0) / neg
    frr = np.cumsum(labels == 1) / pos
    diff = np.abs(far - frr)
    e_idx = np.argmin(diff)
    eer = (far[e_idx] + frr[e_idx]) / 2.0 * 100.0
    # minDCF (P_target=0.01)
    dcf = 0.01 * frr + 0.99 * far
    return float(eer), float(scores[e_idx]), float(dcf.min())


def centroid(embs: List[np.ndarray]) -> np.ndarray:
    c = np.mean(embs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


# ─── DB access ─────────────────────────────────────────────────────────────

def get_db_voiceprints(db_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {model_name: {speaker_id: (embedding, seg_count)}}."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT model_name, speaker_type, speaker_id, embedding, segment_count "
        "FROM speaker_voiceprints"
    ).fetchall()
    conn.close()
    result: Dict[str, Dict[str, Any]] = {}
    for model_name, spk_type, spk_id, blob, segs in rows:
        result.setdefault(model_name, {})
        key = spk_id  # Use just the speaker ID as key
        if blob:
            emb = np.frombuffer(blob, dtype=np.float32)
            nrm = np.linalg.norm(emb)
            result[model_name][key] = {
                "embedding": emb / nrm if nrm > 0 else emb,
                "segments": segs,
                "type": spk_type,
            }
    return result


# ─── Collect segments & assign ground-truth speaker labels ─────────────────

def collect_segments() -> List[Tuple[str, Path]]:
    """Return [(speaker_name, wav_path)] based on recording-naming convention.

    The preprocessed directories are named like:
       collection/2022-04-20/田如兰-2204201033/
    The Chinese name before the first hyphen is the customer.
    All segments in that directory belong to both the agent (000) and that customer.
    """
    segs = []
    for date_dir in sorted(PREPROCESSED_ROOT.glob("collection/*/")):
        for call_dir in sorted(date_dir.iterdir()):
            if not call_dir.is_dir():
                continue
            speaker = call_dir.name.split("-")[0]
            for wav in sorted(call_dir.glob("_seg*.wav")):
                segs.append((speaker, wav))
            for wav in sorted(call_dir.glob("*_seg*.wav")):
                segs.append((speaker, wav))
    # Deduplicate
    seen = set()
    unique = []
    for spk, wav in segs:
        if str(wav) not in seen:
            seen.add(str(wav))
            unique.append((spk, wav))
    return unique


# ─── Score / label segments within a call ───────────────────────────────────

def score_segments_vs_agent(
    seg_files: List[Path],
    extractor: OnnxExtractor,
    agent_ref: np.ndarray,
) -> List[Dict[str, Any]]:
    """Score all segments in a call against the agent reference.

    Returns per-segment: idx, file, dur, similarity, embedding.
    """
    results = []
    for i, f in enumerate(seg_files):
        emb = extractor.extract(f)
        dur = 0.0
        if emb is not None:
            import soundfile as sf
            try:
                d, sr = sf.read(str(f))
                dur = len(d) / sr
            except Exception:
                pass
            sim = float(np.dot(emb, agent_ref))
        else:
            sim = None
        results.append({
            "idx": i,
            "file": f.name,
            "emb": emb,
            "sim": sim,
            "dur": round(dur, 2),
        })
    return results


def find_agent_threshold(
    all_scores: np.ndarray,
    min_threshold: float = 0.3,
    max_threshold: float = 0.8,
    n_bins: int = 200,
) -> float:
    """
    Find the best threshold to separate agent vs customer by looking for
    the largest gap in the similarity score distribution.

    策略：对得分排序，找到 score 变化最快的区间（梯度最大的点）。
    这个点通常是坐席-客户分界。
    """
    if len(all_scores) < 5:
        return 0.55  # default
    scores = np.sort(all_scores)
    # Find where the cumulative distribution jumps the most
    diffs = np.diff(scores)
    if len(diffs) == 0:
        return 0.55
    # Smooth with moving window
    window = max(1, len(diffs) // 20)
    if window > 1:
        kernel = np.ones(window) / window
        diffs = np.convolve(diffs, kernel, mode="same")
    max_jump = np.argmax(diffs)
    threshold = scores[max_jump]
    # Clamp
    return float(np.clip(threshold, min_threshold, max_threshold))


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  增量训练对比实验 v2 — 自适应阈值")
    print("  对比旧版（DB 声纹）vs 新版（全量标注段 centroid）的验证得分")
    print("=" * 72)

    # ── Collect segments ────────────────────────────────────────────────
    segments = collect_segments()
    speaker_names = sorted(set(s for s, _ in segments))
    # Group by recording (call directory)
    call_groups: Dict[str, List[Path]] = defaultdict(list)
    for spk, wav_path in segments:
        call_groups[str(wav_path.parent)].append(wav_path)

    print(f"\n  共 {len(segments)} 个 VAD 段, {len(speaker_names)} 个客户: {speaker_names}")
    print(f"  共 {len(call_groups)} 个通话录音")

    # Load DB voiceprints for all models
    db_all = get_db_voiceprints(str(DB_PATH))

    # ── Per-model analysis ──────────────────────────────────────────────
    overall_old = {"eer": 100.0, "model": ""}
    overall_new = {"eer": 100.0, "model": ""}

    for cfg in MODEL_CONFIGS:
        mname = cfg["name"]
        onnx_path = cfg["onnx_path"]
        dim = cfg["embedding_dim"]

        if not onnx_path.exists():
            print(f"\n  ❌ {mname}: ONNX not found")
            continue

        t0 = time.time()
        extractor = OnnxExtractor(onnx_path)

        # Get DB agent reference first
        db_vp = db_all.get(mname, {})
        agent_key = "000"
        agent_ref_data = db_vp.get(agent_key)
        if agent_ref_data is None:
            for k, v in db_vp.items():
                if v.get("type") == "agent":
                    agent_key = k
                    agent_ref_data = v
                    break
        if agent_ref_data is None:
            print(f"  ❌ {mname}: DB 中无坐席声纹")
            continue
        agent_ref = agent_ref_data["embedding"]
        print(f"  DB 坐席声纹: agent_id={agent_key}, {agent_ref_data['segments']} 段")

        # Step A: Extract embeddings and score vs agent reference simultaneously
        logger.info("[%s] 提取 %d 段 embedding...", mname, len(segments))
        per_call_embs: Dict[str, List[Dict]] = {}
        for call_dir, files in call_groups.items():
            per_call_embs[call_dir] = score_segments_vs_agent(files, extractor, agent_ref)

        # ── Step B: Adaptive threshold ──────────────────────────────────
        all_sims = np.array([
            r["sim"] for rrs in per_call_embs.values()
            for r in rrs if r["sim"] is not None
        ])
        if len(all_sims) < 5:
            print(f"  ⚠ {mname}: 有效得分不足 ({len(all_sims)})")
            continue

        # Find threshold from distribution gap
        threshold = find_agent_threshold(all_sims)
        print(f"  自适应坐席阈值: {threshold:.4f} "
              f"(得分范围: {all_sims.min():.4f}–{all_sims.max():.4f})")

        # ── Step C: Label segments ──────────────────────────────────────
        all_segs: List[Dict] = []
        for call_dir, results in per_call_embs.items():
            call_name = Path(call_dir).name
            customer = call_name.split("-")[0]
            for r in results:
                label = "agent" if (r["sim"] is not None and r["sim"] >= threshold) else "customer"
                all_segs.append({
                    **r,
                    "call": call_name,
                    "customer": customer,
                    "label": label,
                })

        n_agent = sum(1 for r in all_segs if r["label"] == "agent")
        n_cust = sum(1 for r in all_segs if r["label"] == "customer")
        n_valid = sum(1 for r in all_segs if r["emb"] is not None)
        print(f"  标注: 坐席={n_agent} 客户={n_cust} (embedding 有效={n_valid})")

        # ── Step D: Build speaker-level embedding groups ────────────────
        agent_embs = [r["emb"] for r in all_segs if r["label"] == "agent" and r["emb"] is not None]
        customer_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
        for r in all_segs:
            if r["label"] == "customer" and r["emb"] is not None:
                customer_groups[r["customer"]].append(r["emb"])

        print(f"  坐席段数: {len(agent_embs)}")
        for spk, embs in sorted(customer_groups.items()):
            print(f"  客户 {spk}: {len(embs)} 段")

        # ── Step E: Build NEW enrollment centroids ──────────────────────
        new_enroll: Dict[str, np.ndarray] = {}
        new_enroll_counts: Dict[str, int] = {}
        if agent_embs:
            new_enroll["000"] = centroid(agent_embs)
            new_enroll_counts["000"] = len(agent_embs)
        for spk, embs in customer_groups.items():
            new_enroll[spk] = centroid(embs)
            new_enroll_counts[spk] = len(embs)

        # OLD enrollment: from DB
        old_enroll: Dict[str, np.ndarray] = {}
        for spk_id, data in db_vp.items():
            old_enroll[spk_id] = data["embedding"]

        # ── Step F: Build test set ──────────────────────────────────────
        # For each speaker, collect all their segments' embeddings
        test_embs: Dict[str, List[np.ndarray]] = {}
        if agent_embs:
            test_embs["000"] = agent_embs
        for spk, embs in customer_groups.items():
            test_embs[spk] = embs

        # ── Step G: Score trials ────────────────────────────────────────
        def trial_scoring(
            enroll: Dict[str, np.ndarray],
            tests: Dict[str, List[np.ndarray]],
        ) -> Tuple[np.ndarray, np.ndarray]:
            scores, labels = [], []
            for spk_en, emb_en in enroll.items():
                for spk_te, embs_te in tests.items():
                    same = spk_en == spk_te
                    for te in embs_te:
                        scores.append(float(np.dot(emb_en, te)))
                        labels.append(1 if same else 0)
            return np.array(scores), np.array(labels)

        # Old trials
        old_scores, old_labels = trial_scoring(old_enroll, test_embs)
        old_eer, old_thr, old_dcf = compute_eer(old_scores, old_labels)

        # New trials
        new_scores, new_labels = trial_scoring(new_enroll, test_embs)
        new_eer, new_thr, new_dcf = compute_eer(new_scores, new_labels)

        # ── Step H: Print results ───────────────────────────────────────
        print(f"\n  {'=' * 52}")
        print(f"  📊 {mname} — 对比结果 (threshold={threshold:.3f})")
        print(f"  {'=' * 52}")
        print(f"  {'指标':<22} {'旧版 (DB)':<16} {'新版 (centroid)':<16}")
        print(f"  {'─' * 54}")
        print(f"  {'Enroll 说话人':<22} {len(old_enroll):<16} {len(new_enroll):<16}")
        print(f"  {'Trials':<22} {len(old_labels):<16} {len(new_labels):<16}")
        print(f"  {'同人/异人':<22} {int(sum(old_labels))}/{int(sum(old_labels==0)):<14}"
              f"{int(sum(new_labels))}/{int(sum(new_labels==0))}")
        print(f"  {'EER (%)':<22} {old_eer:<10.3f}%  {new_eer:<10.3f}%")
        print(f"  {'minDCF@0.01':<22} {old_dcf:<16.4f} {new_dcf:<16.4f}")

        delta = old_eer - new_eer
        arrow = "↑" if delta > 1e-3 else ("↓" if delta < -1e-3 else "→")
        print(f"  {'EER 变化':<22} {'':16} {arrow}{abs(delta):.3f}%")

        # Score distribution
        def dist_info(prefix, sc, lb):
            pos = sc[lb == 1]
            neg = sc[lb == 0]
            sep = float(np.mean(pos)) - float(np.mean(neg)) if len(pos) and len(neg) else 0.0
            pos_str = f"同人 μ={np.mean(pos):.4f} σ={np.std(pos):.4f}" if len(pos) else "同人 N/A"
            neg_str = f"异人 μ={np.mean(neg):.4f} σ={np.std(neg):.4f}" if len(neg) else "异人 N/A"
            print(f"    {prefix}: {pos_str}  |  {neg_str}  |  间隔={sep:.4f}")

        print(f"\n  得分分布 (cosine):")
        dist_info("旧版", old_scores, old_labels)
        dist_info("新版", new_scores, new_labels)

        # Update overall best
        if old_eer < overall_old["eer"]:
            overall_old = {"eer": old_eer, "model": mname}
        if new_eer < overall_new["eer"]:
            overall_new = {"eer": new_eer, "model": mname}

        # ── Step I: PLDA (if enough data) ───────────────────────────────
        if len(new_enroll) >= 3 and sum(len(v) for v in test_embs.values()) >= 10:
            try:
                plda_scores, plda_labels = train_and_score_plda(
                    test_embs, new_enroll, new_enroll_counts, dim
                )
                plda_eer, plda_thr, plda_dcf = compute_eer(plda_scores, plda_labels)
                print(f"\n  🧪 PLDA Backend:")
                print(f"    EER={plda_eer:.3f}%  minDCF={plda_dcf:.4f}")
            except Exception as e:
                logger.warning("PLDA 跳过: %s", e)

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}s")

    # Final summary
    print(f"\n{'=' * 72}")
    print(f"  汇总")
    print(f"  {'─' * 30}")
    print(f"  旧版最低 EER: {overall_old['model']} ({overall_old['eer']:.3f}%)")
    print(f"  新版最低 EER: {overall_new['model']} ({overall_new['eer']:.3f}%)")
    print(f"{'=' * 72}")


# ─── PLDA (standalone) ─────────────────────────────────────────────────────

M_LOG_2PI = 1.8378770664093455


def train_and_score_plda(
    test_embs: Dict[str, List[np.ndarray]],
    enroll: Dict[str, np.ndarray],
    enroll_counts: Dict[str, int],
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train PLDA on test_embs, then score enroll vs test trials."""

    class _CI:
        def __init__(self, w, n, m):
            self.weight = w
            self.num_example = n
            self.mean = m.reshape(-1, 1) if m.ndim == 1 else m

    # Build stats
    num_classes = 0
    class_weight = 0.0
    example_weight = 0.0
    sum_vec = np.zeros((dim, 1))
    offset_scatter = np.zeros((dim, dim))
    classinfo = []

    for spk, embs in test_embs.items():
        if len(embs) < 1:
            continue
        group = np.array(embs)
        n = group.shape[0]
        mean = np.mean(group, axis=0).reshape((-1, 1))
        weight = 1.0
        offset_scatter += weight * (group.T @ group)
        offset_scatter += -n * weight * (mean @ mean.T)
        classinfo.append(_CI(weight, n, mean))
        num_classes += 1
        class_weight += weight
        example_weight += weight * n
        sum_vec += weight * mean

    if num_classes < 2:
        raise ValueError("Too few speakers for PLDA")

    # EM estimation
    within_var = np.eye(dim)
    between_var = np.eye(dim)

    for iteration in range(15):
        within_var_stats = np.zeros((dim, dim))
        within_var_count = 0.0
        between_var_stats = np.zeros((dim, dim))
        between_var_count = 0.0

        within_var_stats += offset_scatter
        within_var_count += example_weight - class_weight

        within_var_inv = np.linalg.inv(within_var)
        between_var_inv = np.linalg.inv(between_var)
        overall_mean = sum_vec / class_weight

        for ci in classinfo:
            n = ci.num_example
            w = ci.weight
            if n == 0:
                continue
            mix_var = np.linalg.inv(between_var_inv + n * within_var_inv)
            m = (ci.mean - overall_mean).reshape((-1, 1))
            temp = n * (within_var_inv @ m)
            w_vec = mix_var @ temp
            m_w = m - w_vec
            between_var_stats += w * mix_var
            between_var_stats += w * (w_vec @ w_vec.T)
            between_var_count += w
            within_var_stats += w * n * mix_var
            within_var_stats += w * n * (m_w @ m_w.T)
            within_var_count += w

        within_var = within_var_stats / within_var_count
        between_var = between_var_stats / between_var_count

    # Build scoring transform
    c_inv = np.linalg.inv(np.linalg.cholesky(within_var))
    between_proj = c_inv @ between_var @ c_inv.T
    s, U = np.linalg.eigh(between_proj)
    idx = np.argsort(s)[::-1]
    s, U = s[idx], U[:, idx]
    transform = U.T @ c_inv
    psi = s

    mean = overall_mean.reshape(-1)

    def plda_score(enroll_vec, test_vec, n_enroll):
        offset = -transform @ mean
        e_tr = transform @ enroll_vec + offset
        t_tr = transform @ test_vec + offset
        # Length norm
        inv_covar = psi + 1.0 / n_enroll
        dot = np.dot(inv_covar, e_tr ** 2)
        norm_e = np.sqrt(dim / dot) if dot > 0 else 1.0
        e_tr *= norm_e
        dot = np.dot(psi + 1.0, t_tr ** 2)
        norm_t = np.sqrt(dim / dot) if dot > 0 else 1.0
        t_tr *= norm_t
        # LLR
        mean_same = (n_enroll * psi / (n_enroll * psi + 1.0)) * e_tr
        var_same = 1.0 + psi / (n_enroll * psi + 1.0)
        sqdiff = (t_tr - mean_same) ** 2
        loglike_given = -0.5 * (np.sum(np.log(var_same)) + M_LOG_2PI * dim + np.sum(sqdiff / var_same))
        var_total = psi + 1.0
        sqdiff = t_tr ** 2
        loglike_total = -0.5 * (np.sum(np.log(var_total)) + M_LOG_2PI * dim + np.sum(sqdiff / var_total))
        return float(loglike_given - loglike_total)

    # Score trials
    scores, labels = [], []
    for spk_en, emb_en in enroll.items():
        n_en = enroll_counts.get(spk_en, 1)
        for spk_te, embs_te in test_embs.items():
            same = spk_en == spk_te
            for te in embs_te:
                scores.append(plda_score(emb_en, te, n_en))
                labels.append(1 if same else 0)

    return np.array(scores), np.array(labels)


if __name__ == "__main__":
    main()
