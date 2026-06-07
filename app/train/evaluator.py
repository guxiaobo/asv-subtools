"""
模型评估模块。

在测试集上计算 EER / minDCF 等指标，
并与历史版本对比判断是否改进。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("train.evaluator")

# ---------------------------------------------------------------------------
# EER (Equal Error Rate) calculation
# ---------------------------------------------------------------------------


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    计算等错误率 (EER) 和对应阈值。

    Args:
        scores: 相似度分数（越大越可能是同一人）。
        labels: 标签（1=同一人, 0=不同人）。

    Returns:
        (eer, threshold) — EER 值（百分比）和对应阈值。
    """
    if len(scores) == 0 or len(labels) == 0:
        return 1.0, 0.5

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    # Sort by score descending
    sort_idx = np.argsort(scores)[::-1]
    scores_sorted = scores[sort_idx]
    labels_sorted = labels[sort_idx]

    # Genuine and impostor counts
    pos_count = np.sum(labels_sorted == 1)
    neg_count = np.sum(labels_sorted == 0)

    if pos_count == 0 or neg_count == 0:
        return 1.0, 0.5

    # FAR and FRR at each threshold
    far = 1.0 - np.cumsum(labels_sorted == 0) / neg_count
    frr = np.cumsum(labels_sorted == 1) / pos_count

    # Find EER (where FAR ≈ FRR)
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    eer = (far[idx] + frr[idx]) / 2.0 * 100.0

    return float(eer), float(scores_sorted[idx])


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.01,
) -> Tuple[float, float]:
    """
    计算最小检测代价函数 (minDCF)。

    Args:
        scores: 相似度分数。
        labels: 标签（1=同一人, 0=不同人）。
        c_miss: 漏报代价。
        c_fa: 误报代价。
        p_target: 目标先验概率。

    Returns:
        (min_dcf, threshold) — 最小 DCF 值和对应阈值。
    """
    if len(scores) == 0 or len(labels) == 0:
        return 1.0, 0.5

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    sort_idx = np.argsort(scores)[::-1]
    scores_sorted = scores[sort_idx]
    labels_sorted = labels[sort_idx]

    pos_count = np.sum(labels_sorted == 1)
    neg_count = np.sum(labels_sorted == 0)

    if pos_count == 0 or neg_count == 0:
        return 1.0, 0.5

    far = 1.0 - np.cumsum(labels_sorted == 0) / neg_count
    frr = np.cumsum(labels_sorted == 1) / pos_count

    # DCF = C_miss × P_target × FRR + C_fa × (1 - P_target) × FAR
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    min_idx = np.argmin(dcf)

    return float(dcf[min_idx]), float(scores_sorted[min_idx])


# ---------------------------------------------------------------------------
# Scoring comparison
# ---------------------------------------------------------------------------


def should_publish(
    new_eer: float,
    prev_eer: Optional[float],
    improvement_threshold: float = 0.001,
) -> Tuple[bool, float]:
    """
    判断新模型是否应该发布。

    Args:
        new_eer: 新模型的 EER。
        prev_eer: 上一版本模型的 EER（None 表示首版）。
        improvement_threshold: 提升绝对阈值（EER 降低至少达到此值才发布）。

    Returns:
        (should_publish, improvement) — 是否发布，提升幅度（正值表示更好的改进）。
    """
    if prev_eer is None:
        # 首版模型直接发布
        return True, 0.0

    improvement = prev_eer - new_eer
    if improvement > improvement_threshold:
        return True, improvement
    else:
        logger.info(
            "模型无显著改进: prev EER=%.3f%% new EER=%.3f%% (threshold=%.4f)",
            prev_eer, new_eer, improvement_threshold,
        )
        return False, improvement


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------


def evaluate_on_test_set(
    model_path: str,
    test_set_path: str,
    eval_config: Optional[Dict] = None,
) -> Dict:
    """
    在测试集上评估模型性能。

    这是一个骨架方法，实际评估需要 ASV-Subtools 的 extract + scoring 流程。

    Args:
        model_path: ONNX 模型文件路径。
        test_set_path: 测试集目录（包含 enrollment 和 trial 数据）。
        eval_config: 评估配置。

    Returns:
        评估结果字典，包含 eer, min_dcf, threshold 等。
    """
    logger.info("Evaluating model: %s on test set: %s", model_path, test_set_path)

    # TODO: 接入 ASV-Subtools 的 scoring pipeline
    # 1. 使用 ONNX 模型提取每个试音的 embedding
    # 2. 计算成对相似度
    # 3. 与 trial 标签对比计算 EER/minDCF
    #
    # 示例:

    eer, threshold = compute_eer(
        scores=np.array([0.9, 0.8, 0.3, 0.2]),
        labels=np.array([1, 1, 0, 0]),
    )

    min_dcf, dcf_threshold = compute_min_dcf(
        scores=np.array([0.9, 0.8, 0.3, 0.2]),
        labels=np.array([1, 1, 0, 0]),
    )

    return {
        "eer": round(eer, 3),
        "eer_threshold": round(threshold, 4),
        "min_dcf": round(min_dcf, 4),
        "min_dcf_threshold": round(dcf_threshold, 4),
        "test_samples": 4,
        "genuine_trials": 2,
        "impostor_trials": 2,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_eval_summary(
    eval_result: Dict,
    prev_eval_result: Optional[Dict] = None,
) -> str:
    """格式化评估结果为可读字符串。"""
    lines = [
        "========= 模型评估 =========",
        f"  测试样本数: {eval_result.get('test_samples', '?')}",
        f"  同人试音: {eval_result.get('genuine_trials', '?')}",
        f"  异人试音: {eval_result.get('impostor_trials', '?')}",
        f"  EER: {eval_result.get('eer', '?'):.3f}%",
        f"  EER Threshold: {eval_result.get('eer_threshold', '?'):.4f}",
        f"  minDCF: {eval_result.get('min_dcf', '?'):.4f}",
    ]

    if prev_eval_result:
        prev_eer = prev_eval_result.get("eer", 0)
        curr_eer = eval_result.get("eer", 0)
        diff = prev_eer - curr_eer
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        lines.append(f"  相较于上一版本: {prev_eer:.3f}% → {curr_eer:.3f}% ({arrow}{diff:.3f}%)")

    lines.append("==========================")
    return "\n".join(lines)
