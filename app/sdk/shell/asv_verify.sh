#!/usr/bin/env bash
#
# asv_verify.sh — Shell SDK for ASV Speaker Verification API
#
# A bash client using curl + jq for the ASV verification service.
#
# Usage:
#   export ASV_API_URL="http://localhost:8000"
#
#   # Mode A: direct file upload
#   ./asv_verify.sh verify-files audio_a.wav audio_b.wav \
#       --scenario debt_collection --threshold 0.7
#
#   # Mode B: indirect by audio ID
#   ./asv_verify.sh verify-ids recording-001 recording-002 \
#       --backend-a nas --backend-b s3 \
#       --scenario customer_service
#
#   # Health check
#   ./asv_verify.sh health
#
# Requires: curl, jq

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASV_API_URL="${ASV_API_URL:-http://localhost:8000}"
ASV_API_KEY="${ASV_API_KEY:-}"
ASV_TIMEOUT="${ASV_TIMEOUT:-30}"
VERSION="0.1.0"

# Colors (for terminal output)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
ASV Shell SDK v${VERSION} — Speaker Verification API client

Usage:
  $(basename "$0") <command> [options]

Commands:
  verify-files <audio_a> <audio_b> [options]
        Verify two speakers by uploading audio files.
        Options: --scenario <str>, --threshold <0-1>,
                 --scoring-method <cosine|euclidean|dot_product>

  verify-ids <id_a> <id_b> [options]
        Verify two speakers by audio ID.
        Options: --backend-a <nas|s3|redis>, --backend-b <nas|s3|redis>,
                 --scenario <str>, --threshold <0-1>,
                 --scoring-method <str>,
                 --bucket-a <str>, --bucket-b <str>

  health
        Query API health status.

  batch <json_file>
        Run multiple verifications from a JSON file.

Environment:
  ASV_API_URL       API base URL (default: http://localhost:8000)
  ASV_API_KEY       API key (optional, sent as Bearer token)
  ASV_TIMEOUT       Request timeout in seconds (default: 30)

Examples:
  $(basename "$0") verify-files ./voice_a.wav ./voice_b.wav --scenario debt_collection
  $(basename "$0") verify-ids abc-123 def-456 --backend-a nas --threshold 0.7
  $(basename "$0") health
EOF
    exit 0
}

die() {
    echo -e "${RED}ERROR:${NC} $*" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}WARN:${NC} $*" >&2
}

info() {
    echo -e "${GREEN}$*${NC}"
}

check_deps() {
    command -v curl >/dev/null 2>&1 || die "curl is required but not installed."
    command -v jq   >/dev/null 2>&1 || die "jq is required but not installed."
}

# Build curl args common to all requests
curl_args() {
    local args=(
        -sS
        --max-time "$ASV_TIMEOUT"
    )
    if [[ -n "$ASV_API_KEY" ]]; then
        args+=(-H "Authorization: Bearer ${ASV_API_KEY}")
    fi
    echo "${args[@]}"
}

# Make a request and parse JSON response.
# Returns the raw JSON, exits with error code on failure.
do_request() {
    local method="$1"
    local url="$2"
    shift 2

    local response
    local http_code
    local temp_file
    temp_file=$(mktemp)
    trap 'rm -f "$temp_file"' RETURN

    # shellcheck disable=SC2046
    http_code=$(curl $(curl_args) -X "$method" \
        -w '%{http_code}' \
        -o "$temp_file" \
        "$@" \
        "${ASV_API_URL}${url}" 2>/dev/null || true)

    response=$(cat "$temp_file")

    # Check curl exit code
    if [[ -z "$http_code" ]]; then
        die "Network error: curl failed to connect to ${ASV_API_URL}${url}"
    fi

    # Validate JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        die "Invalid JSON response (HTTP ${http_code}): ${response}"
    fi

    # Check HTTP status
    if [[ "$http_code" -ge 400 ]]; then
        local error_msg
        error_msg=$(echo "$response" | jq -r '.error.message // .error // .detail // "Unknown error"' 2>/dev/null)
        die "Server error (HTTP ${http_code}): ${error_msg}"
    fi

    echo "$response"
}

# ---------------------------------------------------------------------------
# Command: health
# ---------------------------------------------------------------------------
cmd_health() {
    check_deps
    info "Checking ASV API health at ${ASV_API_URL}..."

    local response
    response=$(do_request GET "/health")

    local status model_loaded model_path uptime cache
    status=$(echo "$response" | jq -r '.status')
    model_loaded=$(echo "$response" | jq -r '.model_loaded')
    model_path=$(echo "$response" | jq -r '.model_path')
    uptime=$(echo "$response" | jq -r '.uptime_sec')
    cache=$(echo "$response" | jq -r '.cache_connected')

    echo ""
    echo "Status:       ${status}"
    echo "Model loaded: ${model_loaded}"
    echo "Model path:   ${model_path}"
    echo "Uptime:       ${uptime}s"
    echo "Cache:        ${cache}"
    echo ""
    info "Health check complete."
}

# ---------------------------------------------------------------------------
# Command: verify-files
# ---------------------------------------------------------------------------
cmd_verify_files() {
    check_deps
    local audio_a="$1"
    local audio_b="$2"
    shift 2

    # Parse options
    local scenario=""
    local threshold=""
    local scoring_method=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)           scenario="$2";           shift 2 ;;
            --threshold)          threshold="$2";          shift 2 ;;
            --scoring-method)     scoring_method="$2";     shift 2 ;;
            *) die "Unknown option: $1. Use --help." ;;
        esac
    done

    # Validate files
    [[ -f "$audio_a" ]] || die "File not found: ${audio_a}"
    [[ -f "$audio_b" ]] || die "File not found: ${audio_b}"

    info "Verifying: ${audio_a} <-> ${audio_b}"

    # Build curl -F arguments
    local form_args=(
        -F "audio_a=@${audio_a}"
        -F "audio_b=@${audio_b}"
    )
    [[ -n "$scenario" ]]       && form_args+=(-F "scenario=${scenario}")
    [[ -n "$threshold" ]]      && form_args+=(-F "threshold=${threshold}")
    [[ -n "$scoring_method" ]] && form_args+=(-F "scoring_method=${scoring_method}")

    local response
    response=$(do_request POST "/api/verify" "${form_args[@]}")

    print_verify_result "$response"
}

# ---------------------------------------------------------------------------
# Command: verify-ids
# ---------------------------------------------------------------------------
cmd_verify_ids() {
    check_deps
    local audio_id_a="$1"
    local audio_id_b="$2"
    shift 2

    local backend_a="nas"
    local backend_b="nas"
    local scenario=""
    local threshold=""
    local scoring_method=""
    local bucket_a=""
    local bucket_b=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backend-a)       backend_a="$2";        shift 2 ;;
            --backend-b)       backend_b="$2";        shift 2 ;;
            --scenario)        scenario="$2";         shift 2 ;;
            --threshold)       threshold="$2";        shift 2 ;;
            --scoring-method)  scoring_method="$2";   shift 2 ;;
            --bucket-a)        bucket_a="$2";         shift 2 ;;
            --bucket-b)        bucket_b="$2";         shift 2 ;;
            *) die "Unknown option: $1. Use --help." ;;
        esac
    done

    info "Verifying IDs: ${audio_id_a} <-> ${audio_id_b}"

    # Build JSON payload
    local payload
    payload=$(jq -n \
        --arg id_a "$audio_id_a" \
        --arg id_b "$audio_id_b" \
        --arg backend_a "$backend_a" \
        --arg backend_b "$backend_b" \
        --arg scenario "$scenario" \
        --arg threshold "$threshold" \
        --arg scoring_method "$scoring_method" \
        --arg bucket_a "$bucket_a" \
        --arg bucket_b "$bucket_b" \
        '{
            mode: "indirect",
            audio_a: { audio_id: $id_a, storage_backend: $backend_a },
            audio_b: { audio_id: $id_b, storage_backend: $backend_b }
        }
        | if $scenario != ""         then . + {scenario: $scenario} else . end
        | if $threshold != ""        then . + {threshold: ($threshold | tonumber)} else . end
        | if $scoring_method != ""   then . + {scoring_method: $scoring_method} else . end
        | if $bucket_a != ""         then .audio_a.bucket = $bucket_a else . end
        | if $bucket_b != ""         then .audio_b.bucket = $bucket_b else . end
        ')

    local response
    response=$(do_request POST "/api/verify/indirect" \
        -H "Content-Type: application/json" \
        -d "$payload")

    print_verify_result "$response"
}

# ---------------------------------------------------------------------------
# Command: batch
# ---------------------------------------------------------------------------
cmd_batch() {
    check_deps
    local json_file="$1"
    [[ -f "$json_file" ]] || die "Batch file not found: ${json_file}"

    info "Running batch verification from: ${json_file}"

    local total passed failed
    total=$(jq 'length' "$json_file")
    passed=0
    failed=0

    for i in $(seq 0 $((total - 1))); do
        local item
        item=$(jq ".[$i]" "$json_file")
        local mode
        mode=$(echo "$item" | jq -r '.mode // "files"')

        echo ""
        info "[$((i+1))/${total}] Running verification..."

        local response=""
        if [[ "$mode" == "ids" ]]; then
            local id_a id_b backend_a backend_b
            id_a=$(echo "$item" | jq -r '.audio_id_a')
            id_b=$(echo "$item" | jq -r '.audio_id_b')
            backend_a=$(echo "$item" | jq -r '.backend_a // "nas"')
            backend_b=$(echo "$item" | jq -r '.backend_b // "nas"')

            response=$(cmd_verify_ids_inner "$id_a" "$id_b" \
                --backend-a "$backend_a" --backend-b "$backend_b" \
                --scenario "$(echo "$item" | jq -r '.scenario // ""')" \
                --threshold "$(echo "$item" | jq -r '.threshold // ""')") || true
        else
            local file_a file_b
            file_a=$(echo "$item" | jq -r '.audio_a')
            file_b=$(echo "$item" | jq -r '.audio_b')

            response=$(cmd_verify_files_inner "$file_a" "$file_b" \
                --scenario "$(echo "$item" | jq -r '.scenario // ""')" \
                --threshold "$(echo "$item" | jq -r '.threshold // ""')") || true
        fi

        if [[ -n "$response" ]] && echo "$response" | jq -e '.success == true' >/dev/null 2>&1; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo ""
    info "Batch complete: ${total} total, ${passed} passed, ${failed} failed"
}

# Inner versions that return JSON instead of printing
cmd_verify_files_inner() {
    local audio_a="$1"
    local audio_b="$2"
    shift 2

    local form_args=(-F "audio_a=@${audio_a}" -F "audio_b=@${audio_b}")
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)       form_args+=(-F "scenario=${2}");       shift 2 ;;
            --threshold)      form_args+=(-F "threshold=${2}");      shift 2 ;;
            --scoring-method) form_args+=(-F "scoring_method=${2}"); shift 2 ;;
            *) shift ;;
        esac
    done
    do_request POST "/api/verify" "${form_args[@]}" 2>/dev/null || echo ""
}

cmd_verify_ids_inner() {
    local audio_id_a="$1"
    local audio_id_b="$2"
    shift 2

    local backend_a="nas"
    local backend_b="nas"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backend-a)  backend_a="$2";  shift 2 ;;
            --backend-b)  backend_b="$2";  shift 2 ;;
            --scenario)   scenario="$2";   shift 2 ;;
            --threshold)  threshold="$2";  shift 2 ;;
            *) shift ;;
        esac
    done

    local payload
    payload=$(jq -n \
        --arg id_a "$audio_id_a" \
        --arg id_b "$audio_id_b" \
        --arg backend_a "$backend_a" \
        --arg backend_b "$backend_b" \
        '{mode: "indirect",
          audio_a: {audio_id: $id_a, storage_backend: $backend_a},
          audio_b: {audio_id: $id_b, storage_backend: $backend_b}}')
    do_request POST "/api/verify/indirect" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo ""
}

# ---------------------------------------------------------------------------
# Print verification result as formatted table
# ---------------------------------------------------------------------------
print_verify_result() {
    local response="$1"

    local success same_speaker score threshold time_ms source_a source_b error scenario

    success=$(echo "$response" | jq -r '.success')
    same_speaker=$(echo "$response" | jq -r '.is_same_speaker')
    score=$(echo "$response" | jq -r '.score')
    threshold=$(echo "$response" | jq -r '.threshold_used')
    time_ms=$(echo "$response" | jq -r '.processing_time_ms')
    source_a=$(echo "$response" | jq -r '.embedding_a.source // "N/A"')
    source_b=$(echo "$response" | jq -r '.embedding_b.source // "N/A"')
    scenario=$(echo "$response" | jq -r '.scenario // "default"')
    error=$(echo "$response" | jq -r '.error // ""')

    echo ""
    echo "═══════════════════════════════════════"
    echo "  ASV Verification Result"
    echo "═══════════════════════════════════════"
    echo "  Scenario:        ${scenario}"
    echo "  Same speaker:    $( [[ "${same_speaker}" == "true" ]] && echo "${GREEN}YES${NC}" || echo "${RED}NO${NC}" )"
    echo "  Score:           ${score}"
    echo "  Threshold:       ${threshold}"
    echo "  Processing:      ${time_ms}ms"
    echo "  Embedding A:     ${source_a}"
    echo "  Embedding B:     ${source_b}"
    echo "───────────────────────────────────────"

    if [[ "$success" != "true" ]]; then
        echo -e "  ${RED}Error: ${error}${NC}"
    fi

    if [[ "$success" == "true" ]]; then
        # Visual indicator
        local bar_width=50
        local filled
        filled=$(echo "$score * $bar_width" | bc -l 2>/dev/null | cut -d. -f1)
        filled=${filled:-0}
        local empty=$((bar_width - filled))
        if [[ $filled -gt $bar_width ]]; then filled=$bar_width; fi
        if [[ $empty -lt 0 ]]; then empty=0; fi

        echo ""
        printf "  Score bar:  "
        printf "${GREEN}%*s${NC}" "$filled" "" | tr ' ' '█'
        printf "${RED}%*s${NC}" "$empty" "" | tr ' ' '░'
        echo ""
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    [[ $# -eq 0 ]] && usage

    local cmd="$1"
    shift

    case "$cmd" in
        verify-files)
            [[ $# -lt 2 ]] && die "Usage: $(basename "$0") verify-files <audio_a> <audio_b> [options]"
            cmd_verify_files "$@"
            ;;
        verify-ids)
            [[ $# -lt 2 ]] && die "Usage: $(basename "$0") verify-ids <id_a> <id_b> [options]"
            cmd_verify_ids "$@"
            ;;
        health)
            cmd_health
            ;;
        batch)
            [[ $# -lt 1 ]] && die "Usage: $(basename "$0") batch <json_file>"
            cmd_batch "$1"
            ;;
        --help|-h)
            usage
            ;;
        *)
            die "Unknown command: ${cmd}. Use --help for usage."
            ;;
    esac
}

main "$@"
