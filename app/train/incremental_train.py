#!/usr/bin/env python3
"""
模型增量训练 CLI 入口。

从 SQLite 数据库读取已完成预处理但尚未训练的录音，
执行增量训练、评估，并根据效果决定是否发布。

Usage:
    python -m train.incremental_train                    # 触发一轮增量训练
    python -m train.incremental_train --dry-run           # 仅检查数据量
    python -m train.incremental_train --force             # 即使无新增也重训
    python -m train.incremental_train --epochs 5          # 覆盖默认 epoch
    python -m train.incremental_train --lr 5e-5           # 覆盖学习率
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from train.db import (
    claim_pending_recordings,
    count_pending_train,
    get_active_model,
    get_connection,
    get_ready_for_training,
    init_db,
    recover_stuck_tasks,
    single_instance_lock,
    update_train_status,
)
from train.config import load_config
from train.trainer import IncrementalTrainer
from train.evaluator import evaluate_on_test_set, format_eval_summary, should_publish
from train.model_manager import (
    export_onnx,
    get_next_version,
    publish_model,
    register_model_version,
)
from train.schemas import TrainSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train.incremental_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模型增量训练模块 — 增量训练 → 评估 → 发布",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅检查可用数据量，不执行训练",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="即使无新增数据也执行训练（重新训练整个模型）",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖训练 epoch 数",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="覆盖学习率",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="SQLite 数据库路径",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="仅导出 ONNX（跳过训练），用于手动重训后发布",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="跳过单实例锁",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    # Load config
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)
    train_cfg = cfg["incremental_train"]
    model_cfg = cfg["model"]

    # Override from CLI
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.lr:
        train_cfg["base_lr"] = args.lr

    # DB connection
    db_path = args.db or cfg["db_path"]
    conn = get_connection(db_path)
    init_db(conn)

    # Recover stuck tasks
    recovered = recover_stuck_tasks(conn)
    if recovered:
        logger.info("已恢复 %d 条卡住的任务", recovered)

    # Single instance lock
    if not args.no_lock and not single_instance_lock("incremental_train"):
        sys.exit(1)

    # Check pending data
    pending_count = count_pending_train(conn)
    logger.info("待训练的录音数: %d", pending_count)

    if pending_count == 0 and not args.force:
        logger.info("无新增待训练录音，跳过本轮训练")
        conn.close()
        return

    if args.dry_run:
        logger.info("可用训练数据: %d 条录音", pending_count)
        conn.close()
        return

    # ──── STEP 1: Claim & prepare training data ────────────────────────────
    rows = get_ready_for_training(conn, limit=-1)  # no limit
    logger.info("准备训练数据: %d 条录音", len(rows))

    # Count speakers
    agent_ids = set()
    for r in rows:
        if r.get("agent_id"):
            agent_ids.add(r["agent_id"])
    num_new_speakers = len(agent_ids)
    logger.info("共 %d 个坐席(说话人)", num_new_speakers)

    if num_new_speakers == 0 and not args.force:
        logger.info("无有效说话人数据，跳过")
        conn.close()
        return

    # Get current model info
    active_model = get_active_model(conn)
    prev_eer = active_model["eval_value"] if active_model else None
    checkpoint_path = model_cfg.get("checkpoint_path", "")
    if not checkpoint_path and active_model:
        checkpoint_path = active_model["model_path"]
    logger.info("当前模型版本: %s (EER=%.3f%%)",
                active_model["version"] if active_model else "无",
                prev_eer or 0)

    # ──── STEP 2: Run incremental training ─────────────────────────────────
    trainer = IncrementalTrainer(
        checkpoint_path=checkpoint_path,
        backbone=model_cfg.get("backbone", "CAM++"),
        embedding_dim=model_cfg.get("embedding_dim", 192),
        config=train_cfg,
    )

    train_result = trainer.train(
        train_data_dir="data/train_incremental",  # TODO: use actual Kaldi data dir
        num_new_speakers=num_new_speakers,
    )
    logger.info("训练统计: %s", train_result)

    # ──── STEP 3: Evaluate ──────────────────────────────────────────────────
    api_models_dir = model_cfg.get("api_models_dir", "")
    import tempfile
    tmp_checkpoint = str(Path(tempfile.gettempdir()) / "asv_incremental_checkpoint.pt")
    trainer.save_checkpoint(tmp_checkpoint)

    eval_result = evaluate_on_test_set(
        model_path=tmp_checkpoint,
        test_set_path=cfg.get("test_set_path", "data/test_set"),
    )
    logger.info(format_eval_summary(eval_result, active_model))

    new_eer = eval_result.get("eer", 100.0)
    should_pub, improvement = should_publish(
        new_eer=new_eer,
        prev_eer=prev_eer,
        improvement_threshold=train_cfg.get("improvement_threshold", 0.001),
    )

    if not should_pub and not args.force:
        logger.info("模型未改进 (improvement=%.4f < threshold)，跳过发布", improvement)
        # Mark all recordings as processed (trained but no publish)
        for row in rows:
            update_train_status(
                conn, row["id"], "done",
                train_result={"trained": True, "published": False,
                              "eer": new_eer, "prev_eer": prev_eer},
                model_version=active_model["version"] if active_model else None,
            )
        conn.close()
        return

    # ──── STEP 4: Export ONNX & publish ────────────────────────────────────
    version = get_next_version(conn)
    onnx_path = str(Path(api_models_dir) / f"{version}.onnx")

    onnx_exported = export_onnx(
        model_checkpoint_path=tmp_checkpoint,
        output_path=onnx_path,
        embedding_dim=model_cfg.get("embedding_dim", 192),
    )

    published_path = publish_model(
        onnx_path=onnx_exported,
        api_models_dir=api_models_dir,
        version=version,
    )

    # ──── STEP 5: Register in SQLite ───────────────────────────────────────
    register_model_version(
        db_path,
        model_name=model_cfg.get("backbone", "CAM++"),
        version=version,
        eval_metric="EER",
        eval_value=new_eer,
        prev_eval_value=prev_eer,
        improved=True,
        train_recording_count=len(rows),
        train_speaker_count=num_new_speakers,
        train_time_sec=train_result.get("duration_sec", 0),
        model_path=published_path,
        notes=f"Incremental training: {len(rows)} recordings, {num_new_speakers} speakers",
    )

    # Mark recordings as trained
    for row in rows:
        update_train_status(
            conn, row["id"], "done",
            train_result={"trained": True, "published": True,
                          "model_version": version,
                          "eer": new_eer, "prev_eer": prev_eer},
            model_version=version,
        )

    # ──── Summary ──────────────────────────────────────────────────────────
    total_duration = time.time() - start_time
    summary = TrainSummary(
        total_pending=len(rows),
        total_agents=num_new_speakers,
        total_segments=sum(1 for _ in rows),
        new_model_version=version,
        new_eer=new_eer,
        prev_eer=prev_eer,
        improved=True,
        published=True,
        duration_sec=round(total_duration, 1),
    )

    logger.info(
        "==================================================\n"
        "增量训练完成:\n"
        "  录音数: %d | 说话人数: %d\n"
        "  新版本: %s | EER: %.3f%% (prev: %s)\n"
        "  模型路径: %s\n"
        "  总耗时: %.1f 秒\n"
        "==================================================",
        summary.total_pending, summary.total_agents,
        summary.new_model_version, summary.new_eer,
        f"{prev_eer:.3f}%" if prev_eer else "N/A",
        published_path,
        summary.duration_sec,
    )

    conn.close()


if __name__ == "__main__":
    main()
