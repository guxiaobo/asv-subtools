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
from typing import Optional

from train.db import (
    claim_pending_recordings,
    count_pending_preprocess,
    get_connection,
    init_db,
    recover_stuck_tasks,
    single_instance_lock,
    update_pre_status,
    upsert_voiceprint,
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
    # ── Diarizer options ──
    parser.add_argument(
        "--diarize", action="store_true", default=True,
        help="VAD 后运行说话人标注（默认打开）",
    )
    parser.add_argument(
        "--no-diarize", action="store_true", default=False,
        help="跳过说话人标注",
    )
    parser.add_argument(
        "--diarizer-model", type=str, default="CAM++",
        choices=["CAM++", "ResNet34", "ECAPA"],
        help="声纹模型",
    )
    parser.add_argument(
        "--diarizer-agent-threshold", type=float, default=None,
        help="坐席判定阈值（默认 None=动态检测）",
    )
    parser.add_argument(
        "--diarizer-cluster-threshold", type=float, default=0.55,
        help="客户聚类阈值",
    )
    parser.add_argument(
        "--cross-aggregate-only", action="store_true", default=False,
        help="仅执行跨录音客户声纹聚合（跳过 VAD/diarize Phase 1）",
    )
    return parser.parse_args()


def process_single(
    conn,
    row: dict,
    preprocessed_root: str,
    vad_kwargs: dict,
    diarizer_kwargs: Optional[dict] = None,
) -> int:
    """
    处理单条录音。
    Args:
        diarizer_kwargs: diarizer 参数（传给 preprocess_recording）。
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
        preprocess_kwargs = {"audio_path": audio_path, "output_dir": str(output_dir)}
        preprocess_kwargs.update(vad_kwargs)
        if diarizer_kwargs:
            preprocess_kwargs.update(diarizer_kwargs)
            # Don't double-pass run_diarization if vad_kwargs somehow has it
            preprocess_kwargs.setdefault("run_diarization", True)
        result = preprocess_recording(**preprocess_kwargs)

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


def _extract_customer_id(filename: str) -> str:
    """从录音文件名提取客户 ID（第一个 '-' 前的部分）。

    文件名格式:
        客户名-YYMMDDHHMM         →  返回 "客户名"
        客户名-YYMMDDHHMM-N       →  返回 "客户名"  (N为重复序号)
        电话号码-YYMMDDHHMM       →  返回 "电话号码"
    """
    stem = Path(filename).stem if "." in filename else filename
    # 从左切第一个 '-'，兼容 '顾子阳-2204232038-2' 这类多 '-' 文件名
    parts = stem.split("-", 1)
    if len(parts) == 2:
        return parts[0]
    return stem


def _cross_call_aggregate_phase(
    conn,
    preprocessed_root: str,
    diarizer_model: str = "CAM++",
    diarizer_agent_threshold: Optional[float] = None,
    diarizer_cluster_threshold: float = 0.55,
) -> None:
    """
    Phase 2: 跨同客户多通录音聚合客户声纹。

    思路：
    1. 从 DB 读取所有 pre_status='done' 的录音
    2. 按客户 ID（从 filename 提取）分组
    3. 对每个客户的多通录音，重新扫描 preprocessed 目录找到 VAD 段文件
    4. 对每通录音重新跑 diarizer（逐通标注）
    5. 汇总同客户所有通的非坐席段 → 跨录音聚类 → centroid 入库

    只处理有多通录音的客户或单通但有多个非坐席段的录音。
    """
    from train.diarizer import SpeakerDiarizer

    # 查询所有已完成的录音
    recs = conn.execute(
        "SELECT id, customer_phone, call_id, biz_system, call_timestamp, pre_result "
        "FROM recordings WHERE pre_status='done' ORDER BY call_id"
    ).fetchall()
    if not recs:
        logger.info("跨录音聚合: 无已完成录音，跳过")
        return

    # 按客户 ID 分组（优先使用 customer_phone 字段，回退到 call_id 提取）
    from collections import defaultdict
    by_customer = defaultdict(list)
    for r in recs:
        cust_id = r["customer_phone"] or _extract_customer_id(r["call_id"])
        by_customer[cust_id].append(dict(r))

    logger.info("跨录音聚合: %d 位客户 / %d 通录音", len(by_customer), len(recs))

    # 初始化 diarizer
    diarizer = SpeakerDiarizer(
        model_path=None,
        model_name=diarizer_model,
        agent_threshold=diarizer_agent_threshold,
        cluster_threshold=diarizer_cluster_threshold,
    )

    # 构建 calls_by_customer 结构
    calls_by_customer = {}
    skipped = 0

    for cust_id, rec_list in by_customer.items():
        calls_info = []
        for rec in rec_list:
            # 定位 VAD 段文件目录
            biz = rec["biz_system"] or "collection"
            ts = rec["call_timestamp"] or ""
            date_str = ts[:10] if ts else "unknown"
            call_id = rec["call_id"]
            seg_dir = Path(preprocessed_root) / biz / date_str / call_id
            if not seg_dir.exists():
                # 尝试其他日期目录
                alt_dirs = list(Path(preprocessed_root).rglob(call_id))
                if alt_dirs:
                    seg_dir = alt_dirs[0]
                else:
                    skipped += 1
                    continue

            seg_files = sorted(seg_dir.glob("*_seg*.wav"))
            if not seg_files:
                skipped += 1
                continue

            calls_info.append({
                "call_id": call_id,
                "segment_files": seg_files,
            })

        if calls_info:
            calls_by_customer[cust_id] = calls_info

    if skipped:
        logger.warning("跨录音聚合: 跳过 %d 通录音（段文件不存在）", skipped)

    if not calls_by_customer:
        logger.info("跨录音聚合: 无可处理的录音段")
        return

    # 对每通录音先跑一遍 diarize（逐通标注），收集结果
    for cust_id, calls_info in calls_by_customer.items():
        for call_info in calls_info:
            results = diarizer.diarize(call_info["segment_files"])
            call_info["diarize_results"] = results

    # 跨录音聚合
    aggregated = diarizer.cross_call_aggregate(calls_by_customer)

    # 写入 DB
    registered = 0
    for agg in aggregated:
        upsert_voiceprint(
            conn,
            model_name=diarizer_model,
            speaker_type="customer",
            speaker_id=agg["customer_id"],
            embedding=agg["embedding"],
            segment_count=agg["num_segments"],
            source_call_ids=agg["source_call_ids"],
        )
        registered += 1
        logger.info(
            "已注册客户声纹: %s (%d 段 / %d 通录音)",
            agg["customer_id"], agg["num_segments"], agg["num_calls"],
        )

    logger.info(
        "跨录音聚合完成: %d 位客户 → 注册 %d 条客户声纹",
        len(calls_by_customer), registered,
    )


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
        diarizer_kwargs = {
            "run_diarization": not args.no_diarize,
            "diarizer_model_name": args.diarizer_model,
            "diarizer_agent_threshold": args.diarizer_agent_threshold,
            "diarizer_cluster_threshold": args.diarizer_cluster_threshold,
        }
        process_single(conn, row, cfg["preprocessed_root"], pre_cfg, diarizer_kwargs)
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

    # Diarizer kwargs from CLI args
    diarizer_kwargs = {
        "run_diarization": not args.no_diarize,
        "diarizer_model_name": args.diarizer_model,
        "diarizer_agent_threshold": args.diarizer_agent_threshold,
        "diarizer_cluster_threshold": args.diarizer_cluster_threshold,
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
                process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs)

            # Phase 2: 跨录音聚合（每批处理完后触发）
            if not args.no_diarize:
                _cross_call_aggregate_phase(
                    conn=conn,
                    preprocessed_root=cfg["preprocessed_root"],
                    diarizer_model=args.diarizer_model,
                    diarizer_agent_threshold=args.diarizer_agent_threshold,
                    diarizer_cluster_threshold=args.diarizer_cluster_threshold,
                )

            # Reconnect to refresh stats
            conn.close()
            conn = get_connection(db_path)
            init_db(conn)

        return

    # Cross-aggregate-only mode: 跳过 Phase 1，直接跑 Phase 2
    if args.cross_aggregate_only:
        logger.info("仅执行跨录音聚合模式（跳过 VAD/diarize）")
        _cross_call_aggregate_phase(
            conn=conn,
            preprocessed_root=cfg["preprocessed_root"],
            diarizer_model=args.diarizer_model,
            diarizer_agent_threshold=args.diarizer_agent_threshold,
            diarizer_cluster_threshold=args.diarizer_cluster_threshold,
        )
        conn.close()
        return

    # One-shot mode
    rows = claim_pending_recordings(conn, limit=args.limit or 0)
    logger.info("领取到 %d 条录音进行处理", len(rows))

    success = 0
    failed = 0
    for row in rows:
        rc = process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs)
        if rc == 0:
            success += 1
        else:
            failed += 1

    logger.info("预处理完成: 成功=%d 失败=%d", success, failed)

    # ── Phase 2: 跨同客户多通录音聚合客户声纹 ──
    if success > 0 and not args.no_diarize:
        _cross_call_aggregate_phase(
            conn=conn,
            preprocessed_root=cfg["preprocessed_root"],
            diarizer_model=args.diarizer_model,
            diarizer_agent_threshold=args.diarizer_agent_threshold,
            diarizer_cluster_threshold=args.diarizer_cluster_threshold,
        )

    conn.close()


if __name__ == "__main__":
    main()
