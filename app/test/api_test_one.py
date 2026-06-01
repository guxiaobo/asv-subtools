#!/usr/bin/env python3
"""API test harness - run one model at a time."""
import sys, time, json, os, subprocess
from pathlib import Path
import requests as req

APP = Path(__file__).resolve().parent.parent
CONFIG = APP / "api/conf/config.yaml"
WAV_A = APP / "test_data/public/us_0010.wav"
WAV_B = APP / "test_data/public/us_0011.wav"

MODEL_NAME = sys.argv[1]  # relative to models/
MODEL_DISPLAY = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME

def set_config(path):
    with open(CONFIG) as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = [f"  path: ./models/{path}" if l.strip().startswith('path:') else l for l in lines]
    with open(CONFIG, 'w') as f:
        f.write('\n'.join(new_lines))

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
            return None, "process died"
        try:
            r = req.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                return r.json(), None
        except Exception as e:
            pass
        time.sleep(1)
    return None, "timeout"

def test():
    print(f"MODEL: {MODEL_DISPLAY} ({MODEL_NAME})")
    sys.stdout.flush()
    
    set_config(MODEL_NAME)
    subprocess.run(["kill", "-9"] + [p for p in os.popen("lsof -ti :8000").read().strip().split('\n') if p],
                   capture_output=True)
    time.sleep(1)
    
    proc = start_server()
    health, err = wait_health(proc)
    if err:
        print(f"SERVER_ERROR: {err}")
        return False
    
    print(f"SERVER_OK: model_loaded={health.get('model_loaded')}")
    
    # Cross speaker
    with open(WAV_A, 'rb') as fa, open(WAV_B, 'rb') as fb:
        files = {'audio_a': ('a.wav', fa, 'audio/wav'), 'audio_b': ('b.wav', fb, 'audio/wav')}
        data = {'scenario': 'customer_service'}
        try:
            r = req.post("http://localhost:8000/api/verify", files=files, data=data, timeout=30)
            j = r.json()
            print(f"CROSS: score={j['score']:.4f} same={j['is_same_speaker']} ms={j['processing_time_ms']:.0f}")
            e = j.get('embedding_a', {})
            if e:
                print(f"EMB: dim={e['dimension']} norm={e['norm']:.4f} src={e['source']}")
        except Exception as ex:
            print(f"CROSS_ERROR: {ex}")
            return False
    
    # Same file
    with open(WAV_A, 'rb') as fa:
        files2 = {'audio_a': ('a.wav', fa, 'audio/wav'), 'audio_b': ('a.wav', fa, 'audio/wav')}
        try:
            r2 = req.post("http://localhost:8000/api/verify", files=files2, data=data, timeout=30)
            j2 = r2.json()
            print(f"SAME: score={j2['score']:.4f} same={j2['is_same_speaker']}")
        except Exception as ex:
            print(f"SAME_ERROR: {ex}")
            return False
    
    proc.terminate()
    proc.wait(timeout=5)
    return True

if __name__ == '__main__':
    ok = test()
    sys.exit(0 if ok else 1)
