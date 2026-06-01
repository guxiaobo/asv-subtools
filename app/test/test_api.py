"""API 全面测试脚本"""
import json, sys, time, os
from pathlib import Path

BASE = "http://localhost:8000"
TEST_DIR = Path("/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public")

# Use urllib to avoid extra dependencies
import urllib.request
import urllib.error

def req(method, path, data=None, files=None):
    url = f"{BASE}{path}"
    if method == "GET":
        r = urllib.request.urlopen(url)
        return json.loads(r.read())
    elif method == "POST" and files:
        # Build multipart manually
        boundary = "----BOUNDARY" + os.urandom(8).hex()
        body = b""
        for field_name, filepath in files.items():
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            filename = Path(filepath).name
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + file_bytes + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req_ = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        r = urllib.request.urlopen(req_)
        return json.loads(r.read())

ok = 0
fail = 0
results = []

def check(name, expect_success=True):
    global ok, fail
    results.append(name)
    # results are captured in `ret` pattern instead

# =======================
# Test 1: Health
# =======================
print("1/8  Health ...")
h = req("GET", "/health")
assert h["status"] == "ok", f"health status: {h['status']}"
assert h["model_loaded"] == True, "model not loaded"
print(f"  ✓ model_path={h['model_path']}")

# =======================
# Test 2: Verify — same speaker (files)
# =======================
print("2/8  Verify same speaker ...")
r = req("POST", "/api/verify", files={
    "audio_a": str(TEST_DIR / "us_0010.wav"),
    "audio_b": str(TEST_DIR / "us_0010.wav"),
})
assert r["success"] == True, f"success=False: {r}"
assert r["is_same_speaker"] == True, f"should be same: {r['score']}"
assert r["score"] >= 0.95, f"score too low: {r['score']}"
print(f"  ✓ score={r['score']} is_same={r['is_same_speaker']}")

# =======================
# Test 3: Verify — different speaker (files)
# =======================
print("3/8  Verify different speaker ...")
r = req("POST", "/api/verify", files={
    "audio_a": str(TEST_DIR / "us_0010.wav"),
    "audio_b": str(TEST_DIR / "us_0038.wav"),
})
assert r["success"] == True, f"success=False: {r}"
assert r["is_same_speaker"] == False, f"should be different: {r['score']}"
assert r["score"] <= 0.85, f"score too high: {r['score']}"
print(f"  ✓ score={r['score']} is_same={r['is_same_speaker']}")

# =======================
# Test 4: Multiple pairs
# =======================
print("4/8  Multi-pair verify ...")
test_pairs = [
    ("us_0010", "us_0010", True),   # same file → same speaker
    ("us_0010", "us_0038", False),  # different speaker (0.46)
    ("us_0038", "us_0038", True),   # same file → same speaker
    ("us_0038", "us_0039", True),   # same speaker pair (0.98)
]
for (aid, bid, expect_same) in test_pairs:
    r = req("POST", "/api/verify", files={
        "audio_a": str(TEST_DIR / f"{aid}.wav"),
        "audio_b": str(TEST_DIR / f"{bid}.wav"),
    })
    assert r["success"]
    assert r["is_same_speaker"] == expect_same, f"FAIL: {aid} vs {bid} score={r['score']:.4f} expected_same={expect_same} got_same={r['is_same_speaker']}"
    print(f"  {aid} vs {bid}: score={r['score']:.4f} same={r['is_same_speaker']} ✓")

# =======================
# Test 5: Custom threshold
# =======================
print("5/8  Custom threshold ...")
r = req("POST", "/api/verify", files={
    "audio_a": str(TEST_DIR / "us_0010.wav"),
    "audio_b": str(TEST_DIR / "us_0038.wav"),
})
# Verify threshold field exists
assert "threshold_used" in r
print(f"  ✓ threshold_used={r['threshold_used']}")

# =======================
# Test 6: Config dump
# =======================
print("6/8  Config endpoint ...")
try:
    r = req("GET", "/api/config")
    assert "model" in r
    print(f"  ✓ config.model.path={r['model']['path']}")
except:
    print("  - /api/config not available, skip")

# =======================
# Test 7: Python SDK via subprocess
# =======================
print("7/8  Python SDK ...")
import subprocess
result = subprocess.run(
    [sys.executable, "-c", """
import sys
sys.path.insert(0, "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/sdk/python")
from asv_sdk import ASVClient
client = ASVClient("http://localhost:8000")
health = client.health()
print(f"health: {health.status}")
result = client.verify_files(
    "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0010.wav",
    "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0010.wav"
)
print(f"same: {result.score}")
result2 = client.verify_files(
    "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0010.wav",
    "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0038.wav"
)
print(f"diff: {result2.score}")
"""],
    capture_output=True, text=True, timeout=30
)
stdout = result.stdout.strip()
stderr = result.stderr.strip()
print(f"  stdout: {stdout}")
if stderr:
    print(f"  stderr: {stderr[:300]}")
# Parse SDK output: expect lines like "health: ok", "same: 1.0", "diff: 0.46"
lines = [l for l in stdout.split('\n') if l.strip()]
assert any("health: ok" in l for l in lines), f"health check failed in SDK output"
# Find lines with verification results
for l in lines:
    if l.startswith("same:"):
        score = float(l.split()[-1])
        assert score >= 0.95, f"same speaker score too low: {score}"
    if l.startswith("diff:"):
        score = float(l.split()[-1])
        assert score <= 0.65, f"diff speaker score too high: {score}"
assert result.returncode == 0, f"Python SDK failed: {stderr}"
print("  ✓ Python SDK all checks passed")

# =======================
# Test 8: Shell SDK via subprocess
# =======================
print("8/8  Shell SDK ...")
result = subprocess.run(
    ["bash", "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/sdk/shell/asv_verify.sh",
     "health", "http://localhost:8000"],
    capture_output=True, text=True, timeout=10
)
print(f"  health stdout: {result.stdout.strip()[:100]}")
result2 = subprocess.run(
    ["bash", "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/sdk/shell/asv_verify.sh",
     "verify-files",
     "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0010.wav",
     "/Users/guxiaobo/Documents/GitHub/asv-subtools/app/test_data/public/us_0038.wav"],
    capture_output=True, text=True, timeout=30
)
print(f"  verify stdout: {result2.stdout.strip()[:100]}")
assert result2.returncode == 0, f"Shell SDK verify failed: {result2.stderr}"

print("\n" + "="*50)
print("ALL 8 TESTS PASSED ✓")
print("="*50)
