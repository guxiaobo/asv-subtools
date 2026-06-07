#!/usr/bin/env python3
"""
录音预处理 CLI 入口。

从 SQLite 数据库获取未处理的录音清单，循环执行 VAD 切割、降噪等预处理，
并将结果写回 SQLite。

Usage:
    python -m train.preprocess                        # 处理所有待处理录音
    python -m train.preprocess --limit 100            # 最多处理 100 条
    python -m train.preprocess --biz collection       # 只处理催收录音
    python -m train.preprocess --dry-run              # 仅检查清单
    python -m train.preprocess --watch                # 持续监听模式
    python -m train.preprocess --call-id COL...       # 处理单条
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from train.db import (
    claim_pending_recordings,
    count_pending_preprocess,
    get_connection,
    init_db,
    recover_stuck_tasks,
    single_instance_lock,
    update_pre_status,
)
from train.config import load_config
from train.vad import preprocess_recording, get_output_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train.preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="录音预处理模块 — VAD 切割、降噪、质量筛选",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多处理 N 条录音（0=不限制）",
    )
    parser.add_argument(
        "--biz", choices=["collection", "cs"], default=None,
        help="只处理指定业务系统的录音",
    )
    parser.add_argument(
        "--call-id", type=str, default=None,
        help="处理指定 call_id 的单条录音",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅检查待处理清单，不实际处理",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="持续监听模式：每 60 秒轮询新录音",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="SQLite 数据库路径（默认自动查找）",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="跳过单实例锁（调试用）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径",
    )
    return parser.parse_args()


def process_single(
    conn,
    row: dict,
    preprocessed_root: str,
    vad_kwargs: dict,
) -> int:
    """
    处理单条录音。
    Returns: 0 成功, 1 失败。
    """
    rec_id = row["id"]
    audio_path = row["local_audio_path"]
    biz_system = row["biz_system"]
    call_id = row["call_id"]

    if not audio_path or not Path(audio_path).exists():
        logger.warning("录音文件不存在: id=%d path=%s", rec_id, audio_path)
        update_pre_status(conn, rec_id, "failed", pre_error=f"File not found: {audio_path}")
        return 1

    # Parse date from call_timestamp
    date_str = row["call_timestamp"][:10] if row.get("call_timestamp") else "unknown"

    try:
        output_dir = get_output_paths(
            preprocessed_root=preprocessed_root,
            biz_system=biz_system,
            date_str=date_str,
            call_id=call_id,
        )

        # Remove preprocessed results from output_dir key
        result = preprocess_recording(
            audio_path=audio_path,
            output_dir=str(output_dir),
            **vad_kwargs,
        )

        update_pre_status(conn, rec_id, "done", pre_result=result)
        logger.info(
            "Done: id=%d call=%s segments=%d",
            rec_id, call_id, result["segment_count"],
        )
        return 0

    except Exception as e:
        logger.exception("预处理失败: id=%d call=%s", rec_id, call_id)
        update_pre_status(conn, rec_id, "failed", pre_error=str(e))
        return 1


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)
    pre_cfg = cfg["preprocessing"]

    # DB connection
    db_path = args.db or cfg["db_path"]
    conn = get_connection(db_path)
    init_db(conn)

    # Recover stuck tasks on startup
    recovered = recover_stuck_tasks(conn)
    if recovered:
        logger.info("已恢复 %d 条卡住的任务", recovered)

    # Single instance lock (unless --no-lock)
    if not args.no_lock and not single_instance_lock("preprocess"):
        sys.exit(1)

    # Check pending count
    pending = count_pending_preprocess(conn)
    logger.info("待预处理录音数: %d", pending)

    if args.dry_run:
        print(f"待预处理录音: {pending} 条")
        conn.close()
        return

    # Single call-id mode
    if args.call_id:
        from train.db import get_recording_by_call_id
        row = get_recording_by_call_id(conn, args.call_id)
        if not row:
            logger.error("未找到 call_id=%s", args.call_id)
            sys.exit(1)
        process_single(conn, row, cfg["preprocessed_root"], pre_cfg)
        conn.close()
        return

    # VAD kwargs
    vad_kwargs = {
        "sample_rate": pre_cfg.get("target_sample_rate", 16000),
        "window_ms": pre_cfg.get("vad_window_ms", 30),
        "threshold": pre_cfg.get("vad_threshold", 0.5),
        "min_segment_sec": pre_cfg.get("min_segment_sec", 1.5),
        "max_segment_sec": pre_cfg.get("max_segment_sec", 15.0),
        "filter_leading_sec": pre_cfg.get("filter_leading_sec", 2.0),
        "snr_threshold": pre_cfg.get("snr_threshold", 15.0),
        "channel_separated": False,
        "apply_noise_reduction": True,
    }

    # Watch mode: loop forever
    if args.watch:
        logger.info("进入 WATCH 模式，每 60 秒轮询...")
        while True:
            rows = claim_pending_recordings(conn, limit=args.limit or 50)
            if not rows:
                logger.debug("无新录音，等待 60 秒...")
                time.sleep(60)
                continue

            for row in rows:
                process_single(conn, row, cfg["preprocessed_root"], vad_kwargs)

            # Reconnect to refresh stats
            conn.close()
            conn = get_connection(db_path)
            init_db(conn)

        return

    # One-shot mode
    rows = claim_pending_recordings(conn, limit=args.limit or 0)
    logger.info("领取到 %d 条录音进行处理", len(rows))

    success = 0
    failed = 0
    for row in rows:
        rc = process_single(conn, row, cfg["preprocessed_root"], vad_kwargs)
        if rc == 0:
            success += 1
        else:
            failed += 1

    logger.info("预处理完成: 成功=%d 失败=%d", success, failed)
    conn.close()


if __name__ == "__main__":
    main()
