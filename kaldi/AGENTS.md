# Kaldi Integration Scripts

## OVERVIEW
Modified Kaldi scripts for ASV: steps, utils, sid, plus multitask and patch variants. Provides feature extraction, x-vector training (nnet3), speaker ID, and data processing.

## STRUCTURE
```
kaldi/
├── steps/             # Standard Kaldi steps (nnet2, nnet3, online, cleanup, dict, segmentation, tandem)
│   ├── nnet3/         # nnet3 chain training, TDNN configs, xvector extraction
│   └── libs/          # nnet3 xconfig layers (basic_layers, lstm, gru, convolution, trivial_layers)
├── steps_multitask/   # Multi-task learning variant (mirrors steps/ structure)
├── patch/             # Patched Kaldi steps (mirrors steps/ with C++ command patches)
├── utils/             # Data utilities (data/, lang/, nnet/)
├── sid/               # Speaker ID scripts (xvector extraction, PLDA scoring)
└── patch/             # Patch scripts for compiling extra C++ commands
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Train TDNN/x-vector (nnet3) | `steps/nnet3/tdnn/`, `steps/nnet3/chain/` |
| Xconfig layer definitions | `steps/libs/nnet3/xconfig/` |
| Multi-task training | `steps_multitask/` — near-duplicate of `steps/` with MT modifications |
| Speaker ID | `sid/` |
| Data preparation utilities | `utils/data/`, `utils/lang/` |
| Compile extra C++ commands | `patch/runPatch-*.sh` |
| Generate training plots | `steps/nnet3/report/generate_plots.py` |

## CONVENTIONS
- **Three variants of steps/**: `steps/` (standard), `steps_multitask/` (multi-task), `patch/` (patched C++ commands). All share near-identical structure.
- **Python libs**: `steps/libs/nnet3/` uses `__init__.py` for module imports (non-standard for Kaldi).
- **HACK in `utils/nnet/make_nnet_proto.py`**: Multi-word options connected by underscores — intentional workaround.

## NOTES
- `steps_multitask/` and `patch/steps/` are 90%+ duplicates of `steps/` with specific modifications
- Patch scripts compile custom C++ binaries: `nnet3-compile-xvector-net`, `nnet3-offline-xvector-compute`, MMI-GMM commands
- `utils/nnet/make_phone_lm.py` (886 lines) — largest utility script
- `steps/libs/nnet3/xconfig/gru.py` is 2111 lines — the single largest file in kaldi/
