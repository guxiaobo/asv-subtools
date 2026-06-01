# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-25
**Commit:** 09102b1
**Branch:** master

## OVERVIEW
ASV-Subtools: open-source toolkit for speaker recognition (ASV), language identification (LID), built on PyTorch + Kaldi. Two subsystems: v1 (shell/PyTorch framework at root) and v2 (egrecho, Lightning-based, in `subtools2/`).

## STRUCTURE
```
.
├── pytorch/         # v1 PyTorch training: models, launchers, libs
├── subtools2/       # v2 egrecho package (Lightning-based, pip-installable)
│   └── egrecho/     # Core Python package
├── kaldi/           # Kaldi scripts: steps, utils, sid, patch, steps_multitask
├── score/           # Back-end scoring: PLDA, GMM, SVM, whiten, normalization
├── runtime/         # C++ deployment (cmake, kaldifeat, jit export)
├── recipe/          # Example recipes: voxceleb, cnsrc, olr
├── conf/            # Feature/augmentation/VAD config files
├── bin/             # Utility scripts
├── linux/           # Linux helpers (decode_symbolic_link)
└── *.sh             # Top-level shell scripts (data processing, scoring, features)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a new speaker model (v1) | `pytorch/model/*.py` | Inherit `TopVirtualNnet` from `libs/nnet/framework.py` |
| Run PyTorch training (v1) | `pytorch/launcher/run*.py` | Invoke via `runPytorchLauncher.sh` |
| Add a new model (v2/egrecho) | `subtools2/egrecho/models/` | Lightning-based, see `architecture/` |
| Kaldi feature extraction | `kaldi/utils/`, `makeFeatures.sh` | Depends on installed Kaldi |
| PLDA scoring | `score/pyplda/` | Python PLDA with domain adaptation |
| Score normalization (AS-Norm) | `score/ScoreNormalization.py` | S-Norm, AS-Norm |
| Compute metrics (EER, Cavg, minDCF) | `computeEER.sh`, `computeCavg.py`, `computeMin-t-DCF.py` | |
| Data augmentation | `augmentDataByNoise.sh`, `conf/speech_aug_*.yaml` | Reverb, noise, music, babble |
| Export JIT model | `pytorch/pipeline/`, `runtime/` | LibTorch-based inference |
| Config feature extraction | `conf/sre-*.conf`, `conf/vad-*.conf` | MFCC, fbank, pitch configs |
| Recipes (Voxceleb, CNSRC, OLR) | `recipe/` | Each has `run*.sh` entry |
| Multi-GPU training | `runPytorchLauncher.sh` | DDP (default) or Horovod |
| v2 CLI commands | `subtools2/egrecho_cli/` | `egrecho` command after pip install |

## CONVENTIONS
- **Model Blueprint pattern (v1)**: Each model is a standalone `.py` file. Path + init string define the model. No static imports — dynamic loading via blueprint path.
- **Model class (v1)**: Must inherit `TopVirtualNnet` (`pytorch/libs/nnet/framework.py`) for auto-saving, embedding extraction, step-training.
- **Launcher pattern (v1)**: `pytorch/launcher/run*.py` scripts configure training. Copied to project root, invoked via `runPytorchLauncher.sh`.
- **Kaldi integration**: Expects project at `kaldi/egs/<group>/<project>` (4-level path). `path.sh` resolves `KALDI_ROOT` automatically.
- **Data flow (v1)**: Kaldi extracts features (ark/scp) → PyTorch reads via `kaldi_io` → training → x-vectors written back in Kaldi format.
- **v2 (egrecho)**: Modern Python package. `pip install -e .` from `subtools2/`. Uses PyTorch Lightning. CLI via `egrecho` command.
- **Shell scripts**: All `*.sh` at root are data processing utilities. Use `parse_options.sh` for CLI args.
- **Config files**: `conf/` holds `.conf` (feature extraction) and `.yaml` (augmentation chains).

## ANTI-PATTERNS (THIS PROJECT)
- Do NOT call DDP model directly in validation stage (use non-DDP wrapper) — see `trainer_online.py`, `trainer_online_sam.py`
- Do NOT use `from my_model_py import my_model` (static import) — use blueprint pattern for model loading (v1)
- `subtools2/egrecho/pipeline/speaker_embedding.py`: "DO NOT SUPPORT NOW, can not del state cache"

## UNIQUE STYLES
- Model files use hyphenated names (e.g., `ecapa-tdnn-xvector.py`) AND snake_case names (e.g., `ecapa_tdnn_xvector.py`) — both exist
- `kaldi/steps_multitask/` and `kaldi/patch/` are near-duplicates of `kaldi/steps/` with multi-task/patch modifications
- `subtools2/` has its own git repo (`.git/`) — it's a submodule or separate project embedded here
- v1 uses `libs.support.utils` as the universal utility module (imported almost everywhere in pytorch/)
- Stage-based execution: shell scripts and Python launchers use `--stage` / `--endstage` to resume from checkpoints

## COMMANDS
```bash
# v1 PyTorch training
./runPytorchLauncher.sh pytorch/launcher/runResnetXvector.py --gpu-id=0,1,2,3 --stage=0

# v2 egrecho install & CLI
cd subtools2 && pip install -e .
egrecho -h

# Feature extraction (requires Kaldi)
./makeFeatures.sh --stage 0 --endstage 1

# Scoring
./computeEER.sh <score-file>
./computeCavg.py <trials> <scores>
```

## NOTES
- Python ≥3.8 required. PyTorch ≥1.13 for subtools2/egrecho.
- `path.sh` auto-resolves KALDI_ROOT from 4-level directory structure — project must be at `kaldi/egs/<group>/<project>`.
- `subtools2/` is a separate git repo with its own `.git/` — treat as embedded submodule.
- Kaldi `patch/` directory requires running `runPatch-*.sh` to compile extra C++ commands.
- No test suite in v1. subtools2/ has CI for doc building only (`.github/workflows/make-docs.yml`).
- 105 Python files exceed 500 lines — largest complexity in `kaldi/steps/libs/nnet3/xconfig/` and `subtools2/egrecho/`.
