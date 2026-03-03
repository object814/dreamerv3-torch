#!/bin/bash
# ============================================================================
# DreamerV3 Training Time Analysis - Full Benchmark
# ============================================================================
#
# This script runs a comprehensive benchmark of DreamerV3 training speed
# across different environment counts and parallel/sequential modes.
#
# Usage:
#   cd /path/to/Metaworld/third_party/dreamerv3
#   bash run_dreamer_test.sh
#
# Options (edit below or pass via env vars):
#   TASK          - Metaworld task name     (default: metaworld_pick-place-v3)
#   CONFIG        - Config profile          (default: metaworld_visual_200M_heavy_long)
#   STEPS         - Env steps per config    (default: 1000)
#   ENV_COUNTS    - Space-separated counts  (default: "1 8 32 64")
#   DEVICE        - PyTorch device          (default: cuda:0)
#   OUTPUT_DIR    - Output directory        (default: ./logdir)
# ============================================================================

set -e

# Configuration (override via environment variables)
TASK="${TASK:-metaworld_pick-place-v3}"
CONFIG="${CONFIG:-metaworld_visual_200M_heavy_long}"
STEPS="${STEPS:-1000}"
ENV_COUNTS="${ENV_COUNTS:-1 8 16 48}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-./logdir}"

# Navigate to dreamerv3 directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

OUTPUT_FILE="${RESULT_DIR}/dreamer_time_analysis.txt"
JSON_FILE="${RESULT_DIR}/dreamer_time_analysis.json"

echo "=============================================="
echo "DreamerV3 Training Time Analysis"
echo "=============================================="
echo "  Task:       $TASK"
echo "  Config:     $CONFIG"
echo "  Steps:      $STEPS"
echo "  Env counts: $ENV_COUNTS"
echo "  Device:     $DEVICE"
echo "  Output:     $RESULT_DIR"
echo "=============================================="
echo ""

# Run the benchmark
python dreamer_test.py \
    --task "$TASK" \
    --config "$CONFIG" \
    --steps "$STEPS" \
    --env-counts $ENV_COUNTS \
    --device "$DEVICE" \
    --output "$OUTPUT_FILE"

# The JSON is auto-generated alongside the txt
echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "  Text report: $OUTPUT_FILE"
echo "  JSON data:   ${OUTPUT_FILE%.txt}.json"
echo "=============================================="
