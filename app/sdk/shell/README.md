# ASV Shell SDK

Bash shell client for the ASV Speaker Verification API using `curl` + `jq`.

## Requirements

- `curl` — HTTP requests
- `jq` — JSON parsing (`brew install jq`)

## Quick Start

```bash
# Export the API URL
export ASV_API_URL="http://localhost:8000"

# Mode A: verify by uploading audio files
./asv_verify.sh verify-files /path/to/speaker_a.wav /path/to/speaker_b.wav \
    --scenario debt_collection

# Mode B: verify by audio ID
./asv_verify.sh verify-ids recording-001 recording-002 \
    --backend-a nas --backend-b s3 \
    --scenario customer_service

# Health check
./asv_verify.sh health

# Batch verification from JSON file
./asv_verify.sh batch batch_jobs.json
```

## Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `verify-files` | `<audio_a> <audio_b> [options]` | Upload two audio files |
| `verify-ids` | `<id_a> <id_b> [options]` | Verify by audio ID |
| `health` | — | Query API health |
| `batch` | `<json_file>` | Run multiple verifications |

## Options for `verify-files`

| Option | Type | Description |
|--------|------|-------------|
| `--scenario` | string | `customer_service`, `debt_collection`, `audit` |
| `--threshold` | float | Decision threshold (0.0–1.0) |
| `--scoring-method` | string | `cosine`, `euclidean`, `dot_product` |

## Options for `verify-ids`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--backend-a` | string | `nas` | Storage backend for speaker A |
| `--backend-b` | string | `nas` | Storage backend for speaker B |
| `--scenario` | string | — | Business scenario |
| `--threshold` | float | — | Decision threshold override |
| `--scoring-method` | string | — | Scoring method |
| `--bucket-a` | string | — | S3 bucket for speaker A |
| `--bucket-b` | string | — | S3 bucket for speaker B |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASV_API_URL` | `http://localhost:8000` | API base URL |
| `ASV_API_KEY` | — | API key (Bearer token auth) |
| `ASV_TIMEOUT` | `30` | Request timeout in seconds |

## Batch JSON format

```json
[
  {
    "mode": "files",
    "audio_a": "/path/to/a.wav",
    "audio_b": "/path/to/b.wav",
    "scenario": "audit"
  },
  {
    "mode": "ids",
    "audio_id_a": "id-001",
    "audio_id_b": "id-002",
    "backend_a": "nas",
    "scenario": "debt_collection",
    "threshold": 0.75
  }
]
```

## Example Output

```
ASV Verification Result
═══════════════════════════════════════
  Scenario:        debt_collection
  Same speaker:    NO
  Score:           0.35
  Threshold:       0.70
  Processing:      45.2ms
  Embedding A:     computed
  Embedding B:     cached

  Score bar:  █████████████████░░░░░░░░░░░░░░░░░░░░░░
```
