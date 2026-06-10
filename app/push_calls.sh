#!/bin/zsh
# =============================================================================
# 录音推送脚本 — 将 ~/Downloads/calls/ 中的录音文件通过 REST API
# 推送到 ASV 训练系统
#
# API 会自动从文件名推导以下字段：
#   - customer_phone: 第一个 '-' 前的字符串
#   - call_timestamp: 时间戳 (YYMMDDHHMM → ISO 8601)
#   - call_id: 完整文件名（不含扩展名）
#
# 用法:
#   chmod +x push_calls.sh
#   ./push_calls.sh                    # 推送所有文件
#   ./push_calls.sh --dry-run          # 仅打印要推送的列表，不实际推送
#   ./push_calls.sh --limit 3          # 只推送前 3 条
#   ./push_calls.sh --format mp3       # 只推送 .mp3 文件
# =============================================================================

set -e

CALLS_DIR="/Users/guxiaobo/Downloads/calls"
API_URL="http://localhost:8000/api/v1/recordings/push"
AGENT_ID="000"
BIZ_SYSTEM="collection"
DRY_RUN=false
LIMIT=0
FILE_PATTERN=("*" "")  # default: all known audio extensions

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --format)
            case "$2" in
                awb|AWb|AWB) FILE_PATTERN=("*.awb"); shift 2 ;;
                mp3|MP3)     FILE_PATTERN=("*.mp3"); shift 2 ;;
                *) echo "Unknown format: $2 (use awb, mp3, or omit for all)"
                   exit 1 ;;
            esac
            ;;
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

# Collect all matching files
all_files=()
for ext in awb mp3; do
    for f in "$CALLS_DIR"/*.$ext(N); do
        all_files+=("$f")
   	 done
done

total_files=${#all_files[@]}
echo "共发现 $total_files 个录音文件"
echo ""

count=0
success=0
failed=0

for f in "${all_files[@]}"; do
    count=$((count + 1))

    filename=$(basename "$f")
    ext="${filename##*.}"

    # Check if file is readable and has content
    file_size=$(stat -f%z "$f" 2>/dev/null || echo "0")
    if [[ "$file_size" -lt 100 ]]; then
        echo "  [$count/$total_files] ⏭️ 文件太小 ($file_size bytes): $filename"
        failed=$((failed + 1))
        continue
    fi

    if $DRY_RUN; then
        # Show what the API would extract
        name_part="${filename%.*}"
        cust="${name_part%%-*}"
        echo "  [$count/$total_files] [DRY] $filename"
        echo "        客户ID: $cust | file: $f"
        continue
    fi

    # Progress
    echo -n "  [$count/$total_files] $filename ... "

    # Push via curl — API 自动从文件名提取 customer_phone / call_timestamp / call_id
    http_code=$(curl -s -o /tmp/push_resp_$$.json -w "%{http_code}" \
        -X POST "$API_URL" \
        -F "biz_system=$BIZ_SYSTEM" \
        -F "agent_id=$AGENT_ID" \
        -F "audio_source_type=binary" \
        -F "audio_data=@$f" \
        -F "channel_separated=false" 2>/dev/null)

    if [[ "$http_code" == "200" ]]; then
        rec_id=$(python3 -c "
import json, sys
d=json.load(open('/tmp/push_resp_$$.json'))
data=d.get('data',{})
print(f\"{data.get('recording_id','?')} cust={data.get('customer_phone','?')} ts={data.get('call_timestamp','?')}\")" 2>/dev/null)
        echo "✅ id=$rec_id"
        success=$((success + 1))
    else
        err_msg=$(python3 -c "
import json, sys
d=json.load(open('/tmp/push_resp_$$.json'))
print(d.get('detail',''))" 2>/dev/null || echo "HTTP $http_code")
        echo "❌ HTTP=$http_code $err_msg"
        failed=$((failed + 1))
    fi

    rm -f /tmp/push_resp_$$.json

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
