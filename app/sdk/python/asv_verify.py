#!/usr/bin/env python3
"""
ASV Shell SDK — Speaker Verification API client.

A Python CLI tool for the ASV verification service.

用法:
  export ASV_API_URL="http://localhost:8000"

  # Mode A: direct file upload
  python3 asv_verify.py verify-files audio_a.wav audio_b.wav \\
      --scenario debt_collection --threshold 0.7

  # Mode B: indirect by audio ID
  python3 asv_verify.py verify-ids recording-001 recording-002 \\
      --backend-a nas --backend-b s3 \\
      --scenario customer_service

  # Health check
  python3 asv_verify.py health

  # Batch
  python3 asv_verify.py batch batch.json

跨平台兼容：macOS / Windows / Linux
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ASV_API_URL = os.environ.get("ASV_API_URL", "http://localhost:8000")
ASV_API_KEY = os.environ.get("ASV_API_KEY", "")
ASV_TIMEOUT = int(os.environ.get("ASV_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Colors (ANSI — works on all modern terminals; Windows 10+ cmd/PowerShell)
# ---------------------------------------------------------------------------
def _green(text: str) -> str:
    return f"\033[0;32m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[0;31m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[1;33m{text}\033[0m"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_headers() -> dict[str, str]:
    headers = {
        "User-Agent": f"asv-verify-py/{VERSION}",
    }
    if ASV_API_KEY:
        headers["Authorization"] = f"Bearer {ASV_API_KEY}"
    return headers


def do_request(
    method: str,
    path: str,
    *,
    data: Optional[bytes] = None,
    content_type: Optional[str] = None,
    form_fields: Optional[dict[str, str]] = None,
    form_files: Optional[dict[str, Path]] = None,
) -> dict[str, Any]:
    """Send HTTP request and parse JSON response.

    Supports both raw JSON payloads and multipart/form-data.
    """
    url = f"{ASV_API_URL.rstrip('/')}{path}"
    headers = _build_headers()

    body = data
    if form_fields or form_files:
        boundary = "----ASVFormBoundary" + hex(id(data or form_fields))[2:16]
        body = _build_multipart(
            fields=form_fields or {},
            files=form_files or {},
            boundary=boundary,
        )
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif content_type:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=ASV_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            err_msg = (
                err_body.get("error", {}).get("message")
                or err_body.get("error")
                or err_body.get("detail", "Unknown error")
            )
        except Exception:
            err_msg = f"HTTP {e.code}"
        sys.exit(_red(f"Server error (HTTP {e.code}): {err_msg}"))
    except urllib.error.URLError as e:
        sys.exit(_red(f"Network error: {e.reason}"))
    except json.JSONDecodeError as e:
        sys.exit(_red(f"Invalid JSON response: {e}"))


def _build_multipart(
    fields: dict[str, str],
    files: dict[str, Path],
    boundary: str,
) -> bytes:
    """构建 multipart/form-data 请求体。跨平台兼容。"""
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


# ---------------------------------------------------------------------------
# Result printer
# ---------------------------------------------------------------------------


def print_verify_result(response: dict[str, Any]) -> None:
    """打印验证结果为格式化表格。"""
    success = response.get("success", False)
    same_speaker = response.get("is_same_speaker", False)
    score = response.get("score")
    threshold = response.get("threshold_used")
    time_ms = response.get("processing_time_ms")
    source_a = response.get("embedding_a", {}).get("source", "N/A")
    source_b = response.get("embedding_b", {}).get("source", "N/A")
    scenario = response.get("scenario", "default")
    error = response.get("error", "")

    speaker_text = _green("YES") if same_speaker else _red("NO")

    print()
    print("═══════════════════════════════════════")
    print("  ASV Verification Result")
    print("═══════════════════════════════════════")
    print(f"  Scenario:        {scenario}")
    print(f"  Same speaker:    {speaker_text}")
    print(f"  Score:           {score}")
    print(f"  Threshold:       {threshold}")
    print(f"  Processing:      {time_ms}ms")
    print(f"  Embedding A:     {source_a}")
    print(f"  Embedding B:     {source_b}")
    print("───────────────────────────────────────")

    if not success and error:
        print(f"  {_red('Error: ' + error)}")

    if success and score is not None:
        bar_width = 50
        filled = max(0, min(bar_width, int(float(score) * bar_width)))
        empty = bar_width - filled
        print()
        print(f"  Score bar:  {_green('█' * filled)}{_red('░' * empty)}")
        print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_health() -> None:
    """查询 API 健康状态。"""
    print(_green(f"Checking ASV API health at {ASV_API_URL}..."))
    response = do_request("GET", "/health")

    status = response.get("status", "unknown")
    model_loaded = response.get("model_loaded", "unknown")
    model_path = response.get("model_path", "unknown")
    uptime = response.get("uptime_sec", "?")
    cache = response.get("cache_connected", "unknown")

    print()
    print(f"Status:       {status}")
    print(f"Model loaded: {model_loaded}")
    print(f"Model path:   {model_path}")
    print(f"Uptime:       {uptime}s")
    print(f"Cache:        {cache}")
    print()
    print(_green("Health check complete."))


def cmd_verify_files(
    audio_a: str,
    audio_b: str,
    scenario: str = "",
    threshold: Optional[float] = None,
    scoring_method: str = "",
) -> None:
    """通过文件上传验证两个说话人。"""
    path_a = Path(audio_a)
    path_b = Path(audio_b)

    if not path_a.exists():
        sys.exit(_red(f"File not found: {audio_a}"))
    if not path_b.exists():
        sys.exit(_red(f"File not found: {audio_b}"))

    print(_green(f"Verifying: {audio_a} <-> {audio_b}"))

    fields: dict[str, str] = {}
    if scenario:
        fields["scenario"] = scenario
    if threshold is not None:
        fields["threshold"] = str(threshold)
    if scoring_method:
        fields["scoring_method"] = scoring_method

    response = do_request(
        "POST",
        "/api/verify",
        form_fields=fields,
        form_files={"audio_a": path_a, "audio_b": path_b},
    )
    print_verify_result(response)


def cmd_verify_ids(
    audio_id_a: str,
    audio_id_b: str,
    backend_a: str = "nas",
    backend_b: str = "nas",
    scenario: str = "",
    threshold: Optional[float] = None,
    scoring_method: str = "",
    bucket_a: str = "",
    bucket_b: str = "",
) -> None:
    """通过音频 ID 验证两个说话人。"""
    print(_green(f"Verifying IDs: {audio_id_a} <-> {audio_id_b}"))

    payload: dict[str, Any] = {
        "mode": "indirect",
        "audio_a": {"audio_id": audio_id_a, "storage_backend": backend_a},
        "audio_b": {"audio_id": audio_id_b, "storage_backend": backend_b},
    }
    if scenario:
        payload["scenario"] = scenario
    if threshold is not None:
        payload["threshold"] = threshold
    if scoring_method:
        payload["scoring_method"] = scoring_method
    if bucket_a:
        payload["audio_a"]["bucket"] = bucket_a
    if bucket_b:
        payload["audio_b"]["bucket"] = bucket_b

    response = do_request(
        "POST",
        "/api/verify/indirect",
        data=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )
    print_verify_result(response)


def cmd_batch(json_file: str) -> None:
    """从 JSON 文件批量验证。"""
    file_path = Path(json_file)
    if not file_path.exists():
        sys.exit(_red(f"Batch file not found: {json_file}"))

    items = json.loads(file_path.read_text("utf-8"))
    total = len(items)
    passed = 0
    failed = 0

    for i, item in enumerate(items, start=1):
        mode = item.get("mode", "files")
        print()
        print(_green(f"[{i}/{total}] Running verification..."))

        try:
            if mode == "ids":
                id_a = item["audio_id_a"]
                id_b = item["audio_id_b"]
                cmd_verify_ids(
                    id_a,
                    id_b,
                    backend_a=item.get("backend_a", "nas"),
                    backend_b=item.get("backend_b", "nas"),
                    scenario=item.get("scenario", ""),
                    threshold=item.get("threshold"),
                )
                passed += 1
            else:
                file_a = item["audio_a"]
                file_b = item["audio_b"]
                cmd_verify_files(
                    file_a,
                    file_b,
                    scenario=item.get("scenario", ""),
                    threshold=item.get("threshold"),
                )
                passed += 1
        except SystemExit:
            failed += 1
        except Exception as e:
            print(_red(f"  Error: {e}"))
            failed += 1

    print()
    print(
        _green(
            f"Batch complete: {total} total, {passed} passed, {failed} failed"
        )
    )


def _usage(prog: str) -> str:
    return f"""ASV Shell SDK v{VERSION} — Speaker Verification API client

Usage:
  {prog} <command> [options]

Commands:
  verify-files <audio_a> <audio_b> [options]
        Verify two speakers by uploading audio files.
        Options: --scenario <str>, --threshold <0-1>,
                 --scoring-method <cosine|euclidean|dot_product>

  verify-ids <id_a> <id_b> [options]
        Verify two speakers by audio ID.
        Options: --backend-a <nas|s3|redis>, --backend-b <nas|s3|redis>,
                 --scenario <str>, --threshold <0-1>,
                 --scoring-method <str>,
                 --bucket-a <str>, --bucket-b <str>

  health
        Query API health status.

  batch <json_file>
        Run multiple verifications from a JSON file.

Environment:
  ASV_API_URL       API base URL (default: http://localhost:8000)
  ASV_API_KEY       API key (optional, sent as Bearer token)
  ASV_TIMEOUT       Request timeout in seconds (default: 30)

Examples:
  {prog} verify-files ./voice_a.wav ./voice_b.wav --scenario debt_collection
  {prog} verify-ids abc-123 def-456 --backend-a nas --threshold 0.7
  {prog} health
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(_usage(os.path.basename(sys.argv[0])))
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd in ("--help", "-h"):
        print(_usage(os.path.basename(sys.argv[0])))
        return

    # Parse per-command options via argparse fragments
    if cmd == "health":
        cmd_health()

    elif cmd == "verify-files":
        if len(args) < 2:
            sys.exit(_red("Usage: verify-files <audio_a> <audio_b> [options]"))
        audio_a = args[0]
        audio_b = args[1]
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", default="")
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--scoring-method", default="")
        opts, _ = parser.parse_known_args(args[2:])
        cmd_verify_files(
            audio_a, audio_b,
            scenario=opts.scenario,
            threshold=opts.threshold,
            scoring_method=opts.scoring_method,
        )

    elif cmd == "verify-ids":
        if len(args) < 2:
            sys.exit(_red("Usage: verify-ids <id_a> <id_b> [options]"))
        id_a = args[0]
        id_b = args[1]
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend-a", default="nas")
        parser.add_argument("--backend-b", default="nas")
        parser.add_argument("--scenario", default="")
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--scoring-method", default="")
        parser.add_argument("--bucket-a", default="")
        parser.add_argument("--bucket-b", default="")
        opts, _ = parser.parse_known_args(args[2:])
        cmd_verify_ids(
            id_a, id_b,
            backend_a=opts.backend_a,
            backend_b=opts.backend_b,
            scenario=opts.scenario,
            threshold=opts.threshold,
            scoring_method=opts.scoring_method,
            bucket_a=opts.bucket_a,
            bucket_b=opts.bucket_b,
        )

    elif cmd == "batch":
        if len(args) < 1:
            sys.exit(_red("Usage: batch <json_file>"))
        cmd_batch(args[0])

    else:
        sys.exit(_red(f"Unknown command: {cmd}. Use --help for usage."))


if __name__ == "__main__":
    main()
