#!/usr/bin/env python3
"""Analyze checkpoint structures, printing all keys with shapes."""
import sys
sys.path.insert(0, '.')
from pathlib import Path

import torch

WD = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = WD / "pytorch_weights"

checkpoints = {
    "CAM++":      WEIGHTS_DIR / "campplus_cn_common.pt",
    "ECAPA":      WEIGHTS_DIR / "avg_model.pt",
    "ResNet34":   WEIGHTS_DIR / "avg_model",  # no extension
}

for name, path in checkpoints.items():
    print(f"\n{'='*70}")
    print(f"{name}: {path}")
    print(f"{'='*70}")
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    # Handle nested dict
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # Remove 'module.' prefix
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    print(f"Total keys: {len(sd)}")
    # Print all keys grouped by prefix
    groups = {}
    for k, v in sd.items():
        prefix = '.'.join(k.split('.')[:2])
        groups.setdefault(prefix, []).append((k, tuple(v.shape)))
    for prefix in sorted(groups.keys()):
        print(f"\n  [{prefix}]")
        for k, shape in groups[prefix]:
            print(f"    {k}: {shape}")
