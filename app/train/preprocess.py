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
from data.database import SegmentRepo
from data.models import AudioSegmentCreate
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
        help="处理指定 call_id 的录音（单条模式）",
    )
    parser.add_argument(
        "--recording-id", type=int, default=None,
        help="处理指定 recording.id 的录音（单条模式）",
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
    parser.add_argument(
        "--batch-id", type=str, default=None,
        help="断句版本号（如 v2/v3），不传则使用当前最新版本 +1",
    )
    parser.add_argument(
        "--min-ignore", type=float, default=None,
        help="自动忽略短于该秒数的片段（0=不启用）",
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
    batch_id: str = "v1",
) -> int:
    """
    处理单条录音。
    Args:
        diarizer_kwargs: diarizer 参数（传给 preprocess_recording）。
        batch_id: 断句版本号，用于多版本保留。
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
        # 取出仅供 process_single 使用的参数（不会传给 preprocess_recording/energy_vad）
        min_ignore = vad_kwargs.pop("min_segment_sec_ignore", 0.0)
        preprocess_kwargs.update(vad_kwargs)
        if diarizer_kwargs:
            preprocess_kwargs.update(diarizer_kwargs)
            # Don't double-pass run_diarization if vad_kwargs somehow has it
            preprocess_kwargs.setdefault("run_diarization", True)
        result = preprocess_recording(**preprocess_kwargs)

        # ── Write segment metadata to audio_segments table ──
        segment_details = result.get("segment_details", [])
        seg_count = len(segment_details)
        if segment_details:
            from data.models import AudioSegmentCreate

            seg_repo = SegmentRepo(conn)
            # 多版本保留：不删除旧版本，新写入采用 batch_id 区分
            seg_entries = [
                AudioSegmentCreate(
                    recording_id=rec_id,
                    segment_index=i,
                    batch_id=batch_id,
                    file_path=sd["file_path"],
                    start_sec=sd["start_sec"],
                    end_sec=sd["end_sec"],
                    duration_sec=sd["duration_sec"],
                )
                for i, sd in enumerate(segment_details)
            ]

            # ── 自动忽略短片段 ──
            if min_ignore > 0:
                ignored_count = 0
                for entry in seg_entries:
                    if entry.duration_sec < min_ignore:
                        entry.is_ignored = 1
                        entry.speaker_type = "ignored"
                        ignored_count += 1
                if ignored_count:
                    logger.info(
                        "自动忽略 %d/%d 段（时长 < %.1f秒）",
                        ignored_count, len(seg_entries), min_ignore,
                    )

            seg_repo.insert_batch([s.model_dump() for s in seg_entries])
        else:
            logger.warning("No segment_details in result (writing zero segments)")

        # VAD 正常跑完但无语音段 → 标记为"无法断句"，与"未断句"区分
        final_status = "unsegmentable" if seg_count == 0 else "done"
        update_pre_status(conn, rec_id, final_status, pre_result=result)
        logger.info(
            "Done: id=%d call=%s segments=%d status=%s",
            rec_id, call_id, seg_count, final_status,
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
        "SELECT id, customer_id, call_id, biz_system, call_timestamp, pre_result "
        "FROM recordings WHERE pre_status='done' ORDER BY call_id"
    ).fetchall()
    if not recs:
        logger.info("跨录音聚合: 无已完成录音，跳过")
        return

    # 按客户 ID 分组（优先使用 customer_id 字段，回退到 call_id 提取）
    from collections import defaultdict
    by_customer = defaultdict(list)
    for r in recs:
        cust_id = r["customer_id"] or _extract_customer_id(r["call_id"])
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
    calls_by_customer: dict = {}
    skipped = 0

    for cust_id, rec_list in by_customer.items():
        calls_info = []
        for rec in rec_list:
            rec_id = rec["id"]
            # Query audio_segments table instead of filesystem glob
            seg_rows = conn.execute(
                "SELECT id, file_path FROM audio_segments "
                "WHERE recording_id = ? AND is_ignored = 0 "
                "ORDER BY segment_index",
                (rec_id,),
            ).fetchall()
            if not seg_rows:
                skipped += 1
                continue

            seg_files = [Path(r["file_path"]) for r in seg_rows]
            calls_info.append({
                "call_id": rec["call_id"],
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
            try:
                results = diarizer.diarize(call_info["segment_files"])
                call_info["diarize_results"] = results
            except Exception as e:
                logger.warning("跨录音聚合阶段 diarize 失败: call=%s %s — 跳过该通",
                               call_info.get("call_id", "?"), e)
                call_info["diarize_results"] = []

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

    # ── Build filtered VAD kwargs ──
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
        "min_segment_sec_ignore": args.min_ignore if args.min_ignore is not None else pre_cfg.get("min_segment_sec_ignore", 0.0),
    }

    # Single recording-id mode (like --call-id but by DB row id)
    if args.recording_id:
        from train.db import get_recording_by_id
        row = get_recording_by_id(conn, args.recording_id)
        if not row:
            logger.error("未找到 recording_id=%s", args.recording_id)
            sys.exit(1)
        # 标记进入 processing（从 reprocessing 转换），界面显示"重新断句中"
        update_pre_status(conn, args.recording_id, "processing")
        diarizer_kwargs = {
            "run_diarization": not args.no_diarize,
            "diarizer_model_name": args.diarizer_model,
        }
        process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs, batch_id=args.batch_id or "v1")
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
        process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs, batch_id=args.batch_id or "v1")
        conn.close()
        return

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
                process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs, batch_id=args.batch_id or "v1")

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

    # One-shot mode (default batch size = 50)
    rows = claim_pending_recordings(conn, limit=args.limit or 50)
    logger.info("领取到 %d 条录音进行处理", len(rows))

    batch_id = args.batch_id or "v1"
    logger.info("本次断句版本: %s", batch_id)

    success = 0
    failed = 0
    for row in rows:
        rc = process_single(conn, row, cfg["preprocessed_root"], vad_kwargs, diarizer_kwargs, batch_id=batch_id)
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
