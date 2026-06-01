# ASV Python SDK

Python client for the ASV Speaker Verification API.

## Installation

```bash
cd sdk/python
pip install -e .
```

Or install directly:

```bash
pip install httpx
```

## Quick Start

```python
from asv_sdk import ASVClient, verify

# Create a client
client = ASVClient(base_url="http://localhost:8000")

# Mode A: Upload two audio files directly
result = client.verify_files(
    audio_a="/path/to/speaker_a.wav",
    audio_b="/path/to/speaker_b.wav",
    scenario="debt_collection",  # optional
)
print(f"Same speaker: {result.is_same_speaker}")
print(f"Similarity score: {result.score:.4f}")
print(f"Processing time: {result.processing_time_ms:.1f}ms")

# Mode B: Verify by audio ID (files stored on NAS/S3/Redis)
result = client.verify_ids(
    audio_id_a="recording-20250101-001",
    audio_id_b="recording-20250101-002",
    backend_a="nas",
    scenario="customer_service",
)

# Quick one-shot (auto-cleanup)
result = verify(
    audio_a="/path/to/a.wav",
    audio_b="/path/to/b.wav",
    scenario="audit",
)

client.close()
```

## API

### `ASVClient(base_url, api_key, timeout, max_retries)`

| Method | Description |
|--------|-------------|
| `verify_files(audio_a, audio_b, ...)` | Upload two files for verification |
| `verify_ids(audio_id_a, audio_id_b, ...)` | Verify by audio ID |
| `verify_batch(comparisons)` | Batch verification (sequential) |
| `health()` | Query server health |

### `VerifyResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Request succeeded |
| `is_same_speaker` | bool | Whether same speaker |
| `score` | float | Similarity score |
| `threshold_used` | float | Decision threshold applied |
| `processing_time_ms` | float | Server processing time |
| `embedding_a/b.source` | str | "computed" or "cached" |
| `embedding_a/b.dimension` | int | Embedding vector dimension |
| `error` | str | Error message (if any) |

## Batch Processing

```python
comparisons = [
    {"mode": "files", "audio_a": "a1.wav", "audio_b": "b1.wav", "scenario": "audit"},
    {"mode": "ids", "audio_id_a": "id2", "audio_id_b": "id3", "backend_a": "nas"},
]
results = client.verify_batch(comparisons)
for r in results:
    print(r.score, r.is_same_speaker)
```
