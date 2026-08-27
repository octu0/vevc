#!/bin/bash
# Milestone 2 Gate 1 自動検証スクリプト
# 使用素材: ~/Downloads/miko1.y4m
# 判定基準: ビット重み付き被覆率によるファイル比削減見込み >= 2.0% (Gate 1 PASS)

set -euo pipefail

INPUT_FILE="$1.y4m"
WORK_DIR="/tmp/vevc_gate1_work"
DUMP_DIR="${WORK_DIR}/dump"
BITSTREAM="${WORK_DIR}/$1_p2.vevc"
DECODED_Y4M="${WORK_DIR}/$1_p2_dec.y4m"

echo "================================================================================"
echo "Milestone 2 (Step 1): DPCM Predictor Ladder & Gate 1 Verification"
echo "================================================================================"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found." >&2
    exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$DUMP_DIR"

echo "[1/4] Building release binaries..."
swift build -c release

echo "[2/4] Running unit test suite (swift test -c release)..."
swift test -c release

echo "[3/4] Encoding miko1.y4m with VEVC_DPCM_DUMP enabled..."
VEVC_DPCM_DUMP="$DUMP_DIR" swift run -c release vevc-enc \
    -i "$INPUT_FILE" \
    -o "$BITSTREAM" \
    -b 500 \
    -profile 2

echo "[4/4] Verifying full 1801-frame decoding via vevc-dec..."
swift run -c release vevc-dec \
    -i "$BITSTREAM" \
    -o "$DECODED_Y4M" \
    -max-layer 2

if [ ! -f "$DUMP_DIR/dpcm_blocks.bin" ]; then
    echo "Error: Dump file $DUMP_DIR/dpcm_blocks.bin was not generated." >&2
    exit 1
fi

FILE_BYTES=$(stat -f%z "$BITSTREAM" 2>/dev/null || stat -c%s "$BITSTREAM")
TOTAL_FILE_BITS=$((FILE_BYTES * 8))

echo ""
echo "================================================================================"
echo "Running Offline Predictor Ladder Evaluation (32 Configurations)..."
echo "================================================================================"

VEVC_DPCM_EVAL_DIR="$DUMP_DIR" VEVC_DPCM_TOTAL_FILE_BITS="$TOTAL_FILE_BITS" \
    swift test -c release --filter DPCMLadderTests/testRealDumpEvaluationIfAvailable

echo "Gate 1 verification script completed successfully."
