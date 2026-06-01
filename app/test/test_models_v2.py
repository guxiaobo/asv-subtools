#!/usr/bin/env python3
"""Test all three ONNX models via API (direct version)."""
import subprocess, sys, time, signal, json, os
from pathlib import Path
import requests as req

APP = Path(__file__).resolve().parent.parent
CONFIG = APP / "api/conf/config.yaml"
WAV_A = APP / "test_data/public/us_0010.wav"
WAV_B = APP / "test_data/public/us_0011.wav"

MODELS = [
    ("campplus.onnx", "CAM++ (中文 192-dim)", "./models/campplus.onnx"),
    ("voxceleb_resnet34_LM.onnx", "ResNet34-LM 256-dim", "./models/voxceleb_resnet34_LM.onnx"),
    ("ecapa-speaker-v1.onnx", "ECAPA-TDNN 192-dim", "./models/ecapa-speaker-v1.onnx"),
]

RESULTS = []

def set_config(path):
    with open(CONFIG) as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('path:'):
            new_lines.append(f"  path: {path}")
        else:
            new_lines.append(line)
    with open(CONFIG, 'w') as f:
        f.write('\n'.join(new_lines))

def kill_server():
    try:
        subprocess.run(["kill", "-9"] + [p for p in os.popen("lsof -ti :8000").read().strip().split('\n') if p],
                      capture_output=True, timeout=3)
    except:
        pass
    time.sleep(1)

def start_server():
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(APP / "api"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def wait_health(proc, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return None
        try:
            r = req.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        time.sleep(1)
    return None

def test_model(model_name, display, model_path):
    print(f"\n{'='*60}")
    print(f"  Test: {display}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")
    
    set_config(model_path)
    kill_server()
    proc = start_server()
    
    health = wait_health(proc)
    if not health:
        print(f"  ❌ SERVER: failed to start")
        return None
    
    print(f"  ✅ Server started: {health.get('model_loaded')}")
    
    # Test 1: Cross speaker
    with open(WAV_A, 'rb') as fa, open(WAV_B, 'rb') as fb:
        files = {'audio_a': ('a.wav', fa, 'audio/wav'), 'audio_b': ('b.wav', fb, 'audio/wav')}
        data = {'scenario': 'customer_service'}
        try:
            r = req.post("http://localhost:8000/api/verify", files=files, data=data, timeout=30)
            j = r.json()
            print(f"  ✅ Cross speaker: score={j['score']:.4f} same={j['is_same_speaker']} ({j['processing_time_ms']:.0f}ms)")
            try:
                e = j['embedding_a']
                print(f"     Embedding: dim={e['dimension']} norm={e['norm']:.4f} src={e['source']}")
            except:
                pass
            verdict = "PASS"
        except Exception as e:
            print(f"  ❌ Cross speaker: {e}")
            verdict = "FAIL"
    
    # Test 2: Same file
    with open(WAV_A, 'rb') as fa:
        files2 = {'audio_a': ('a.wav', fa, 'audio/wav'), 'audio_b': ('a.wav', fa, 'audio/wav')}
        data = {'scenario': 'customer_service'}
        try:
            r2 = req.post("http://localhost:8000/api/verify", files=files2, data=data, timeout=30)
            j2 = r2.json()
            print(f"  ✅ Same file: score={j2['score']:.4f} same={j2['is_same_speaker']}")
        except Exception as e:
            print(f"  ❌ Same file: {e}")
            verdict = "FAIL"
    
    RESULTS.append((display, verdict))
    print(f"  [DONE] {verdict}")
    
    proc.terminate()
    proc.wait(timeout=5)
    return verdict

# ====== MAIN ======
for name, display, path in MODELS:
    test_model(name, display, path)
    time.sleep(1)

# Restore default
set_config("./models/campplus.onnx")

print(f"\n{'='*60}")
print(f"  SUMMARY:")
for name, status in RESULTS:
    print(f"  {'✅' if status=='PASS' else '❌'} {name}")
print(f"{'='*60}")
