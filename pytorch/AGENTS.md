# PyTorch Training Framework (v1)

## OVERVIEW
v1 speaker/embedding model training framework. Models inherit `TopVirtualNnet`, launched via `run*.py` scripts, managed by `runPytorchLauncher.sh`.

## STRUCTURE
```
pytorch/
├── model/           # Speaker model definitions (*.py)
├── launcher/        # Training launch scripts (run*.py)
├── libs/
│   ├── nnet/        # Neural net components: framework, pooling, loss, components, transformer
│   ├── training/    # Trainers: standard, online, online_sam, mt (multi-task); optimizers
│   ├── egs/         # Data loading: kaldi_dataset, processor, speech_augment, signal_processing
│   └── support/     # Utilities: kaldi_io, utils (universal helper module)
├── pipeline/        # Model export (JIT)
└── bin/             # Training binary helpers
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Add new model architecture | `model/*.py` — inherit `libs/nnet/framework.TopVirtualNnet` |
| Create training launcher | `launcher/run*.py` — copy existing, modify model path + hyperparams |
| Add pooling layer | `libs/nnet/pooling.py` (ASP, LDE, MHA, GMHA, MR-MHA) |
| Add loss function | `libs/nnet/loss.py` (Softmax, AM-Softmax, AAM-Softmax, Ring Loss) |
| Add optimizer | `libs/training/optim.py`, `optim_fd.py` (Lookahead, RAdam, Novograd, GC) |
| Modify data loading | `libs/egs/kaldi_dataset.py`, `processor.py` |
| Augmentation pipeline | `libs/egs/speech_augment.py` |
| Kaldi I/O | `libs/support/kaldi_io.py` |
| Universal utilities | `libs/support/utils.py` — imported almost everywhere |
| Export JIT model | `pipeline/` |
| Online training | `libs/training/trainer_online.py`, `trainer_online_sam.py` |

## CONVENTIONS
- **Blueprint pattern**: Model path + init string (e.g., `resnet(40, 1211, loss="AM")`) — no static imports
- **Stage-based launchers**: `--stage=N` / `--endstage=N` for resumable execution. Stage 3 = training, Stage 4 = extraction
- **Trainer types**: `trainer.py` (standard), `trainer_online.py` (online aug), `trainer_fd.py` (feature decomposition), `trainer_mt.py` (multi-task)
- **Model naming**: Files may use hyphens (`ecapa-tdnn-xvector.py`) OR underscores (`ecapa_tdnn_xvector.py`) — check both

## ANTI-PATTERNS
- Do NOT call DDP model in validation — use non-DDP wrapper (documented in `trainer_online.py`)
- Do NOT hardcode model imports — use blueprint path pattern
- `libs/support/utils.py` has a FIXME: "we limit the support here: we allow padding of only the last dimension"

## NOTES
- `libs/support/utils.py` is the most-imported module in the framework
- Multi-GPU: DDP (default via `torch.distributed.launch`) or Horovod
- Online trainers support mixed precision (AMP)
- Transformer models in `libs/nnet/transformer/` (attention, encoder, subsampling)
