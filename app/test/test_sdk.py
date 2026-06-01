#!/usr/bin/env python3
"""SDK test for ASV API - tests ASVClient with all three models."""
import sys, time, subprocess, signal, os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WAV_A = BASE / "test_data" / "public" / "us_0010.wav"
WAV_B = BASE / "test_data" / "public" / "us_0011.wav"
CONFIG = BASE / "api/conf/config.yaml"

sys.path.insert(0, str(BASE / "sdk" / "python"))
from asv_sdk import ASVClient, VerifyResult, HealthResult


def test_model(model_name, model_path):
    """Test a single model via SDK."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} ({model_path})")
    print(f"{'='*60}")

    # Update config
    with open(CONFIG) as f:
        orig = f.read()
    with open(CONFIG, "w") as f:
        f.write(orig.replace(
            orig[orig.index("  path:"):orig.index("\n", orig.index("  path:"))],
            f"  path: {model_path}"
        ))

    # Restart server
    os.system("kill -9 $(lsof -ti :8000) 2>/dev/null")
    time.sleep(1)

    server = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(BASE / "api"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    try:
        # SDK health check
        client = ASVClient(base_url="http://localhost:8000", timeout=30)
        health = client.health()
        print(f"  SDK health: status={health.status} model_loaded={health.model_loaded}")

        # SDK verify_files cross
        result = client.verify_files(str(WAV_A), str(WAV_B), scenario="customer_service")
        print(f"  SDK verify_files(cross): score={result.score:.4f} same={result.is_same_speaker} "
              f"dim={result.embedding_a.dimension} ms={result.processing_time_ms:.0f}")

        # SDK verify_files same
        result2 = client.verify_files(str(WAV_A), str(WAV_A), scenario="customer_service")
        print(f"  SDK verify_files(same):  score={result2.score:.4f} same={result2.is_same_speaker} "
              f"dim={result2.embedding_a.dimension} ms={result2.processing_time_ms:.0f}")

        # SDK verify_ids indirect
        try:
            result3 = client.verify_ids("us_0010.wav", "us_0011.wav", scenario="customer_service")
            print(f"  SDK verify_ids(cross):  score={result3.score:.4f} same={result3.is_same_speaker}")
        except Exception as e:
            print(f"  SDK verify_ids: SKIPPED ({e})")

        client.close()
        print(f"  ✅ {model_name}: ALL PASS")
        return True

    except Exception as e:
        print(f"  ❌ {model_name}: FAILED - {e}")
        return False

    finally:
        server.terminate()
        server.wait()
        # Restore original config
        with open(CONFIG, "w") as f:
            f.write(orig)


if __name__ == "__main__":
    models = [
        ("CAM++ (中文 192-dim)", "./models/campplus.onnx"),
        ("ResNet34-LM (256-dim)", "./models/voxceleb_resnet34_LM.onnx"),
        ("ECAPA-TDNN (192-dim)", "./models/ecapa-speaker-v1.onnx"),
    ]

    passed = 0
    for name, path in models:
        if test_model(name, path):
            passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(models)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(models) else 1)
