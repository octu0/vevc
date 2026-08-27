#!/bin/bash
set -euo pipefail

# ==============================================================================
# Milestone 1: Gate 0 自動検証スクリプト
# DPCM走査位置別ビット会計による後半ビット比率（保持率 1/4 上限）の検証
# ==============================================================================

INPUT_Y4M="${1}.y4m"
OUTPUT_VEVC="/tmp/$1_m1_gate0.vevc"
LOG_FILE="/tmp/$1_m1_dpcm_stats.log"

if [ ! -f "$INPUT_Y4M" ]; then
    echo "エラー: 入力ファイル '$INPUT_Y4M' が存在しません。" >&2
    exit 1
fi

echo "=== Gate 0 計測開始: 入力素材 $INPUT_Y4M ==="
echo "実行コマンド: VEVC_DPCM_STATS=1 swift run -c release vevc-enc -i \"$INPUT_Y4M\" -b 500 -profile 2 -o \"$OUTPUT_VEVC\""

VEVC_DPCM_STATS=1 swift run -c release vevc-enc -i "$INPUT_Y4M" -b 500 -profile 2 -o "$OUTPUT_VEVC" 2> "$LOG_FILE"

echo ""
echo "=== DPCM 統計ログ (stderr) ==="
cat "$LOG_FILE"
echo ""

# K=4 の後半ビット比率 (File に対する割合) を抽出
K4_LINE=$(grep "K= 4" "$LOG_FILE" || grep "K=4" "$LOG_FILE" || true)

if [ -z "$K4_LINE" ]; then
    echo "エラー: ログから K=4 の統計行を抽出できませんでした。" >&2
    exit 1
fi

# 比率の数値を抽出 (例: "12.34% of file" -> "12.34")
K4_RATIO=$(echo "$K4_LINE" | sed -E 's/.* ([0-9.]+)% of file.*/\1/')

echo "抽出された K=4 (保持率 1/4) 後半ビット比率: ${K4_RATIO}%"

# 閾値 5.0% との比較判定 (awk を使用)
IS_PASS=$(awk -v r="$K4_RATIO" 'BEGIN { if (5.0 <= r) print "1"; else print "0"; }')

if [ "$IS_PASS" = "1" ]; then
    echo ""
    echo "================================================================================"
    echo ">>> [GATE 0: PASS] 後半ビット比率 ${K4_RATIO}% >= 5.0%"
    echo ">>> 削減ポテンシャル上限が 5% 以上であることを確認しました。Step 1 へ進みます。"
    echo "================================================================================"
    exit 0
else
    echo ""
    echo "================================================================================"
    echo ">>> [GATE 0: FAIL] 後半ビット比率 ${K4_RATIO}% < 5.0%"
    echo ">>> 削減ポテンシャルが 5% 未満のため、方針中止判断を要します。"
    echo "================================================================================"
    exit 2
fi
