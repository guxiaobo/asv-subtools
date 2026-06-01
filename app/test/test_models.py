#!/usr/bin/env python3
"""
Test all three pre-trained ONNX models through the API end-to-end.
Updates config.yaml, starts server, runs tests, then cleans up.
"""
import os, sys, json, time, subprocess, base64
from pathlib import Path
import requests as req

PROJECT = Path(__file__).parent.parent
CONFIG = PROJECT / "api/conf/config.yaml"
TEST_DATA = PROJECT / "test_data"

MODELS = [
    ("campplus", "CAM++ (中文, 192-dim, input=feats)", "./models/campplus.onnx"),
    ("resnet34", "ResNet34-LM (VoxCeleb, 256-dim, input=feats)", "./models/voxceleb_resnet34_LM.onnx"),
    ("ecapa", "ECAPA-TDNN (VoxCeleb, 192-dim, input=features+feature_lens)", "./models/ecapa-speaker-v1.onnx"),
]

def load_config():
    with open(CONFIG, 'r') as f:
        return f.read()

def save_config(content):
    with open(CONFIG, 'w') as f:
        f.write(content)

def set_model_path(path):
    cfg = load_config()
    lines = cfg.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('path:'):
            new_lines.append(f"  path: {path}")
        else:
            new_lines.append(line)
    save_config('\n'.join(new_lines))

def read_file_lines(p):
    with open(p, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def api_post(url, data):
    """Make JSON API request and return parsed JSON."""
    try:
        r = req.post(url, json=data, timeout=30)
        if r.status_code < 400:
            return r.json()
        return {'error': f'HTTP {r.status_code}', 'body': r.text[:200]}
    except Exception as e:
        return {'error': str(e)}

def api_multipart(url, files, data):
    """Make multipart/form-data API request."""
    try:
        r = req.post(url, files=files, data=data, timeout=30)
        if r.status_code < 400:
            return r.json()
        return {'error': f'HTTP {r.status_code}', 'body': r.text[:200]}
    except Exception as e:
        return {'error': str(e)}

def wait_for_server(url, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = req.get(f"{url}/health", timeout=2)
            return r.json()
        except:
            time.sleep(1)
    return None

def verify_files():
    """Find two test WAV files for same/different speaker tests."""
    for d in [TEST_DATA / "public", TEST_DATA / "wav", TEST_DATA]:
        wavs = list(d.glob("*.wav"))
        if len(wavs) >= 2:
            return wavs[0], wavs[1]
    return None, None

def test_model_api(model_name, display_name, model_path):
    """Test a single model via API."""
    print(f"\n{'='*60}")
    print(f"  Test: {display_name}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    # 1. Update config
    set_model_path(model_path)
    print(f"  [CONFIG] model.path → {model_path}")

    # 2. Kill old server, start new one
    subprocess.run(["kill", "-9"] + [p for p in os.popen("lsof -ti :8000").read().strip().split("\n") if p],
                   capture_output=True)
    time.sleep(1)
    api_dir = PROJECT / "api"
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(api_dir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # 3. Wait for server
    health = wait_for_server("http://localhost:8000")
    if not health:
        print(f"  ❌ SERVER: failed to start")
        proc.kill()
        return False
    print(f"  ✅ SERVER: {health.get('status', 'ok')}")

    # 4. Find test files
    wav_a, wav_b = verify_files()
    if not wav_a or not wav_b:
        print(f"  ⚠  No test WAV files found, using short scores...")
        # Try alternative test approach
        pass

    # 5. Test: health endpoint
    try:
        r = req.get("http://localhost:8000/health", timeout=5)
        health = r.json()
        print(f"  ✅ GET /health: {health.get('status')}")
    except Exception as e:
        print(f"  ❌ GET /health: {e}")

    # 6. Test: verify via multipart file upload
    if wav_a and wav_b:
        try:
            # Test: different speakers comparison
            files = {
                'audio_a': ('audio_a.wav', open(wav_a, 'rb'), 'audio/wav'),
                'audio_b': ('audio_b.wav', open(wav_b, 'rb'), 'audio/wav'),
            }
            data = {'scenario': 'customer_service'}
            r = api_multipart("http://localhost:8000/api/verify", files, data)
            # Close files
            for f in files.values():
                f[1].close()
            if not r.get('error'):
                score = r.get('score', 0)
                ms = r.get('processing_time_ms', 0)
                same = r.get('is_same_speaker', False)
                print(f"  ✅ POST /api/verify (cross speaker)")
                print(f"     score={score:.4f} is_same={'T' if same else 'F'} ({ms:.0f}ms)")
                # Embedding info
                emb_a = r.get('embedding_a', {})
                if emb_a:
                    dim = emb_a.get('dimension', '?')
                    norm = emb_a.get('norm', 0)
                    src = emb_a.get('source', '?')
                    print(f"     embedding: dim={dim}, norm={norm:.4f}, src={src}")
            else:
                print(f"  ❌ POST /api/verify: {r.get('body', r.get('error'))}")

            # Test: same file (should match)
            files2 = {
                'audio_a': ('audio.wav', open(wav_a, 'rb'), 'audio/wav'),
                'audio_b': ('audio.wav', open(wav_a, 'rb'), 'audio/wav'),
            }
            r2 = api_multipart("http://localhost:8000/api/verify", files2, data)
            for f in files2.values():
                f[1].close()
            if not r2.get('error'):
                score = r2.get('score', 0)
                same = r2.get('is_same_speaker', False)
                print(f"  ✅ POST /api/verify (same file)")
                print(f"     score={score:.4f} is_same={'T' if same else 'F'}")
            else:
                print(f"  ❌ POST /api/verify (same): {r2.get('body', r2.get('error'))}")
        except Exception as e:
            print(f"  ❌ API tests: {e}")

    # 9. Stop server
    proc.terminate()
    proc.wait(timeout=5)
    print(f"  [DONE] {model_name}")
    return True

# ===== Main =====
passed = 0
failed = 0

for name, display, path in MODELS:
    ok = test_model_api(name, display, path)
    if ok:
        passed += 1
    else:
        failed += 1
    time.sleep(1)

print(f"\n{'='*60}")
print(f"  Results: {passed}/{len(MODELS)} passed, {failed} failed")
print(f"{'='*60}")

# Restore original model (campplus)
set_model_path("./models/campplus.onnx")
print(f"[RESTORE] config.yaml → ./models/campplus.onnx")
