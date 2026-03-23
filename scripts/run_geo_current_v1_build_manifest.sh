#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/dataset/zsy/dataset-extracted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/geo_current_v1}"
FAMILY_MANIFEST="${FAMILY_MANIFEST:-$OUTPUT_ROOT/family_manifest.jsonl}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SPLITS="${SPLITS:-train val}"

mkdir -p "$OUTPUT_ROOT"

echo "[GeoCurrentV1] build family manifest -> $FAMILY_MANIFEST shard=$((SHARD_INDEX + 1))/$NUM_SHARDS"
"$PYTHON_BIN" "$ROOT/scripts/build_geo_current_family_manifest.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-manifest "$FAMILY_MANIFEST" \
  --splits $SPLITS \
  --tile-size-px "${TILE_SIZE_PX:-896}" \
  --overlap-px "${OVERLAP_PX:-232}" \
  --keep-margin-px "${KEEP_MARGIN_PX:-116}" \
  --review-crop-pad-px "${REVIEW_CROP_PAD_PX:-64}" \
  --tile-min-mask-ratio "${TILE_MIN_MASK_RATIO:-0.02}" \
  --tile-min-mask-pixels "${TILE_MIN_MASK_PIXELS:-256}" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --search-within-review-bbox \
  --fallback-to-all-if-empty
