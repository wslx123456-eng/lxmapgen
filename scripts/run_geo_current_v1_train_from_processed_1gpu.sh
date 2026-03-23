#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_ROOT="${PROCESSED_ROOT:-$ROOT/outputs/geo_current_v1_merged}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$ROOT/outputs/geo_current_v1_train_from_processed}"

if [[ ! -f "$PROCESSED_ROOT/stage_a/dataset/train.jsonl" ]]; then
  echo "[GeoCurrentV1] missing processed Stage A dataset under $PROCESSED_ROOT/stage_a/dataset" >&2
  exit 1
fi
if [[ ! -f "$PROCESSED_ROOT/stage_b/dataset/train.jsonl" ]]; then
  echo "[GeoCurrentV1] missing processed Stage B dataset under $PROCESSED_ROOT/stage_b/dataset" >&2
  exit 1
fi

echo "[GeoCurrentV1] train from processed Stage A"
OUTPUT_ROOT="$TRAIN_OUTPUT_ROOT" \
STAGE_A_ROOT="${STAGE_A_ROOT:-$TRAIN_OUTPUT_ROOT/stage_a}" \
STAGE_A_OUTPUT="${STAGE_A_OUTPUT:-$TRAIN_OUTPUT_ROOT/stage_a/checkpoints}" \
STAGE_A_DATASET="$PROCESSED_ROOT/stage_a/dataset" \
BUILD_MANIFEST_IF_MISSING=0 \
EXPORT_STAGE_A_DATASET=0 \
bash "$ROOT/scripts/run_geo_current_v1_stagea_train_1gpu.sh"

echo "[GeoCurrentV1] train from processed Stage B"
OUTPUT_ROOT="$TRAIN_OUTPUT_ROOT" \
STAGE_A_OUTPUT="${STAGE_A_OUTPUT:-$TRAIN_OUTPUT_ROOT/stage_a/checkpoints}" \
STAGE_B_ROOT="${STAGE_B_ROOT:-$TRAIN_OUTPUT_ROOT/stage_b}" \
STAGE_B_OUTPUT="${STAGE_B_OUTPUT:-$TRAIN_OUTPUT_ROOT/stage_b/checkpoints}" \
STAGE_B_DATASET="$PROCESSED_ROOT/stage_b/dataset" \
BUILD_MANIFEST_IF_MISSING=0 \
EXPORT_STAGE_B_DATASET=0 \
TRAIN_STAGE_A_IF_MISSING=0 \
bash "$ROOT/scripts/run_geo_current_v1_stageb_train_1gpu.sh"
