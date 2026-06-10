#!/usr/bin/env python3
"""Verify all three models can load their checkpoints correctly."""
import sys, torch
sys.path.insert(0, '.')
from train.fine_tune import *

WEIGHTS = Path('pytorch_weights')
device = 'cpu'

def test(name, model_cls, ckpt_path, **kwargs):
    print(f'\n{"="*60}')
    print(f'{name}: loading {ckpt_path.name}')
    print(f'{"="*60}')
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    print(f'  Checkpoint keys: {len(ckpt)}')
    model = model_cls(**kwargs)
    model.load_pretrained(ckpt)
    # Forward test
    B, T, F = 2, 200, 80
    dummy = torch.randn(B, T, F)
    out_shape = model(dummy).shape
    print(f'  Forward OK: dummy={dummy.shape} -> {out_shape}')
    print(f'  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
    return True

results = []
try:
    results.append(test('CAM++', CAMPlus, WEIGHTS/'campplus_cn_common.pt', feat_dim=80, embedding_dim=192))
except Exception as e:
    print(f'  FAILED: {e}')
    results.append(False)

try:
    results.append(test('ECAPA-TDNN', ECAPA_TDNNSpeaker, WEIGHTS/'avg_model.pt', feat_dim=80, embedding_dim=192))
except Exception as e:
    print(f'  FAILED: {e}')
    results.append(False)

try:
    results.append(test('ResNet34 2D', ResNet34_2D, WEIGHTS/'avg_model', feat_dim=80, embedding_dim=256))
except Exception as e:
    print(f'  FAILED: {e}')
    results.append(False)

print(f'\n{">"*60}')
print(f'Results: {sum(results)}/3 models loaded successfully')
for i, (name, ok) in enumerate(zip(['CAM++', 'ECAPA', 'ResNet34'], results)):
    print(f'  {name}: {"PASS" if ok else "FAIL"}')
print(f'{">"*60}')
