"""Verify pretrained weight loading for all three models."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

import torch

def main():
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "pytorch_weights")

    models = [
        ("CAM++", "campp_model", "create_campp_model", "campplus_cn_common.pt", {}),
        ("ResNet34", "resnet34_model", "create_resnet34_model", "avg_model", {}),
        ("ECAPA", "ecapa_model", "create_ecapa_model", "avg_model.pt", {}),
    ]

    for name, module_name, fn_name, weight_file, extra_kwargs in models:
        print(f"=== {name} ===")
        mod = __import__(module_name)
        fn = getattr(mod, fn_name)
        weight_path = os.path.join(weights_dir, weight_file)

        if not os.path.exists(weight_path):
            print(f"  Weight file not found: {weight_path}")
            continue

        print(f"  Loading {weight_path} ({os.path.getsize(weight_path)/1024/1024:.1f} MB)")
        model, status = fn(pretrained_path=weight_path, **extra_kwargs)

        if status["loaded"]:
            print(f"  ✓ ALL WEIGHTS LOADED SUCCESSFULLY")
        else:
            print(f"  ⚠ Partial load:")
            if status.get("missing"):
                print(f"    Missing: {status['missing'][:5]}... ({len(status['missing'])} total)")
            if status.get("unexpected"):
                print(f"    Unexpected: {status['unexpected'][:5]}... ({len(status['unexpected'])} total)")
            if status.get("error"):
                print(f"    Error: {status['error']}")

        # Test forward pass
        model.eval()
        dummy = torch.randn(1, 200, 80)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Forward: input(1,200,80) -> output{tuple(out.shape)}")
        print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
        print()

    # Compare MPS availability
    print(f"=== MPS available: {torch.backends.mps.is_available()} ===")


if __name__ == "__main__":
    main()
