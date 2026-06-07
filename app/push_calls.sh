#!/bin/zsh
# =============================================================================
# 录音推送脚本 — 将 /Users/guxiaobo/Downloads/calls/ 中的 .awb 录音文件
# 通过 REST API 推送到 ASV 训练系统
#
# 用法:
#   chmod +x push_calls.sh
#   ./push_calls.sh                    # 推送所有文件
#   ./push_calls.sh --dry-run          # 仅打印要推送的列表，不实际推送
#   ./push_calls.sh --limit 3          # 只推送前 3 条
# =============================================================================

set -e

CALLS_DIR="/Users/guxiaobo/Downloads/calls"
API_URL="http://localhost:8000/api/v1/recordings/push"
AGENT_ID="000"
BIZ_SYSTEM="collection"
DRY_RUN=false
LIMIT=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================="
echo "  录音推送工具"
echo "  目录: $CALLS_DIR"
echo "  API:  $API_URL"
echo "  坐席: $AGENT_ID"
echo "  业务: $BIZ_SYSTEM"
echo "================================================="
echo ""

# Count total
total_files=0
count=0
success=0
failed=0

for f in "$CALLS_DIR"/*.awb(N); do
    total_files=$((total_files + 1))
done

echo "共发现 $total_files 个录音文件"
echo ""

for f in "$CALLS_DIR"/*.awb(N); do
    count=$((count + 1))
    
    # Extract filename without path and extension
    filename=$(basename "$f" .awb)
    
    # Parse: {name}-{YYMMDDHHMM}[-N]
    # Use zsh regex matching - get the base name and timestamp
    # Timestamp: last group of 10 digits after a hyphen
    if [[ "$filename" =~ ^(.+)-([0-9]{10})(-[0-9]+)?$ ]]; then
        name="${match[1]}"
        ts="${match[2]}"      # YYMMDDHHMM
        seg="${match[3]}"     # Optional: -1, -2 (or empty)
    else
        echo "  [!] 跳过 (格式不匹配): $filename"
        failed=$((failed + 1))
        continue
    fi
    
    # Convert YYMMDDHHMM → ISO 8601
    YY="${ts:0:2}"
    MM="${ts:2:2}"
    DD="${ts:4:2}"
    HH="${ts:6:2}"
    MI="${ts:8:2}"
    
    # Assume YY = 20YY (years 2020-2029)
    isoTimestamp="20${YY}-${MM}-${DD}T${HH}:${MI}:00"
    
    # Unique call_id = full filename (includes segment suffix if any)
    call_id="$filename"
    
    # customer_phone = Chinese name (as customer ID)
    customer_phone="$name"
    
    # Check if file is readable
    file_size=$(stat -f%z "$f" 2>/dev/null || echo "0")
    if [[ "$file_size" -lt 100 ]]; then
        echo "  [!] 文件太小 ($file_size bytes): $(basename $f)"
        failed=$((failed + 1))
        continue
    fi
    
    if $DRY_RUN; then
        echo "  [$count/$total_files] [DRY] $(basename $f)"
        echo "        客户: $customer_phone | 时间: $isoTimestamp | call_id: $call_id"
        continue
    fi
    
    # Progress indicator
    echo -n "  [$count/$total_files] $(basename $f) ... "
    
    # Push via curl
    http_code=$(curl -s -o /tmp/push_resp_$$.json -w "%{http_code}" \
        -X POST "$API_URL" \
        -F "biz_system=$BIZ_SYSTEM" \
        -F "agent_id=$AGENT_ID" \
        -F "customer_phone=$customer_phone" \
        -F "call_timestamp=$isoTimestamp" \
        -F "call_id=$call_id" \
        -F "audio_source_type=binary" \
        -F "audio_data=@$f" \
        -F "channel_separated=false" 2>/dev/null)
    
    if [[ "$http_code" == "200" ]]; then
        rec_id=$(python3 -c "import json; d=json.load(open('/tmp/push_resp_$$.json')); print(d.get('data',{}).get('recording_id','?'))" 2>/dev/null)
        echo "✅ id=$rec_id"
        success=$((success + 1))
    else
        err_msg=$(python3 -c "import json; d=json.load(open('/tmp/push_resp_$$.json')); print(d.get('detail','unknown error'))" 2>/dev/null || echo "HTTP $http_code")
        echo "❌ HTTP=$http_code $err_msg"
        failed=$((failed + 1))
    fi
    
    # Cleanup temp file
    rm -f /tmp/push_resp_$$.json
    
    # Apply limit if set
    if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
        echo ""
        echo "已达到 --limit $LIMIT，停止"
        break
    fi
done

echo ""
echo "================== 结果汇总 =================="
echo "  总计: $total_files"
if ! $DRY_RUN; then
    echo "  成功: $success"
    echo "  失败: $failed"
fi
echo "================================================="
