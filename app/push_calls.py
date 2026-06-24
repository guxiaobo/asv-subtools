#!/usr/bin/env python3
"""
录音推送脚本 — 将 ~/Downloads/calls/ 中的录音文件通过 REST API
推送到 ASV 训练系统。

API 会自动从文件名推导以下字段：
  - customer_id: 第一个 '-' 前的字符串
  - call_timestamp: 时间戳 (YYMMDDHHMM → ISO 8601)
  - call_id: 完整文件名（不含扩展名）

用法:
  python3 push_calls.py                      # 推送所有文件
  python3 push_calls.py --dry-run            # 仅打印要推送的列表，不实际推送
  python3 push_calls.py --limit 3            # 只推送前 3 条
  python3 push_calls.py --format mp3         # 只推送 .mp3 文件
  python3 push_calls.py --calls-dir /path    # 自定义录音文件目录

跨平台兼容：macOS / Windows / Linux
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("push_calls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="录音推送工具 — 将本地录音文件推送至 ASV 训练系统",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印要推送的列表，不实际推送",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多推送 N 条（0=不限制）",
    )
    parser.add_argument(
        "--format", choices=["awb", "mp3"], default=None,
        help="只推送指定格式的文件（默认推送所有支持的格式）",
    )
    parser.add_argument(
        "--calls-dir",
        default=str(Path.home() / "Downloads" / "calls"),
        help="录音文件目录（默认: ~/Downloads/calls）",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/recordings/push",
        help="API URL（默认: http://localhost:8000/api/v1/recordings/push）",
    )
    parser.add_argument(
        "--agent-id", default="000",
        help="坐席 ID（默认: 000）",
    )
    parser.add_argument(
        "--biz-system", default="collection",
        choices=["collection", "cs"],
        help="业务系统（默认: collection）",
    )
    return parser.parse_args()


def scan_audio_files(calls_dir: str, fmt: Optional[str] = None) -> list[Path]:
    """扫描目录中的音频文件，按修改时间排序（最旧优先）。"""
    ext_patterns = [f"*.{fmt}"] if fmt else ["*.awb", "*.mp3"]
    files: list[Path] = []
    for pattern in ext_patterns:
        files.extend(Path(calls_dir).glob(pattern))
    # 去重并排序（最旧优先）
    files = sorted(set(files), key=lambda p: p.stat().st_mtime)
    return files


def _build_multipart(
    fields: dict[str, str],
    files: dict[str, Path],
    boundary: str,
) -> bytes:
    """构建 multipart/form-data 请求体。

    纯标准库实现，不依赖 requests/curl。跨平台兼容。
    """
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(f"{value}\r\n".encode())

    for field, path in files.items():
        filename = path.name
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body)


def push_file(
    file_path: Path,
    api_url: str,
    agent_id: str,
    biz_system: str,
) -> dict:
    """通过 multipart POST 推送单个录音文件。

    Returns:
        API 返回的 JSON 字典。
    """
    boundary = "----ASVPushBoundary" + hex(id(file_path))[2:16]

    body = _build_multipart(
        fields={
            "biz_system": biz_system,
            "agent_id": agent_id,
            "audio_source_type": "binary",
            "channel_separated": "false",
        },
        files={"audio_data": file_path},
        boundary=boundary,
    )

    req = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_body = {"detail": f"HTTP {e.code}"}
        raise RuntimeError(err_body.get("detail", str(err_body))) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络错误: {e.reason}") from e


def format_result(data: dict) -> str:
    """从 API 响应中提取可读的信息。"""
    d = data.get("data", data)
    parts = []
    if "recording_id" in d:
        parts.append(f"id={d['recording_id']}")
    if "customer_id" in d:
        parts.append(f"cust={d['customer_id']}")
    if "call_timestamp" in d:
        parts.append(f"ts={d['call_timestamp']}")
    return " ".join(parts)


def main() -> None:
    args = parse_args()

    print("=" * 52)
    print(f"  录音推送工具")
    print(f"  目录: {args.calls_dir}")
    print(f"  API:  {args.api_url}")
    print(f"  坐席: {args.agent_id}")
    print(f"  业务: {args.biz_system}")
    print("=" * 52)
    print()

    all_files = scan_audio_files(args.calls_dir, args.format)
    if not all_files:
        print("未发现音频文件")
        sys.exit(0)

    total = len(all_files)
    print(f"共发现 {total} 个录音文件")
    print()

    success = 0
    failed = 0

    for i, f in enumerate(all_files, start=1):
        file_size = f.stat().st_size
        if file_size < 100:
            print(f"  [{i}/{total}] ⏭️ 文件太小 ({file_size} bytes): {f.name}")
            failed += 1
            continue

        if args.dry_run:
            cust = f.stem.split("-")[0]
            print(f"  [{i}/{total}] [DRY] {f.name}")
            print(f"        客户ID: {cust} | file: {f}")
            continue

        print(f"  [{i}/{total}] {f.name} ... ", end="", flush=True)
        try:
            data = push_file(f, args.api_url, args.agent_id, args.biz_system)
            print(f"✅ {format_result(data)}")
            success += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

        if args.limit > 0 and (success + failed) >= args.limit:
            print()
            print(f"已达到 --limit {args.limit}，停止")
            break

    print()
    print("=" * 52)
    print(f"  总计文件: {total}")
    if not args.dry_run:
        print(f"  成功推送: {success}")
        print(f"  失败: {failed}")
    print("=" * 52)


if __name__ == "__main__":
    main()
