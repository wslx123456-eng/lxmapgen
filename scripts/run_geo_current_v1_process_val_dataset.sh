#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/dataset/zsy/dataset-extracted}"
VAL_OUTPUT_ROOT="${VAL_OUTPUT_ROOT:-$ROOT/outputs/geo_current_v1_val}"
VAL_FAMILY_MANIFEST="${VAL_FAMILY_MANIFEST:-$VAL_OUTPUT_ROOT/family_manifest_val.jsonl}"
export SPLITS="${SPLITS:-val}"
export SHARD_TAG="${SHARD_TAG:-val_shard_$(printf '%02d' "${SHARD_INDEX:-0}")_of_$(printf '%02d' "${NUM_SHARDS:-1}")}"

mkdir -p "$VAL_OUTPUT_ROOT"

echo "[GeoCurrentV1] build val-only family manifest -> $VAL_FAMILY_MANIFEST shard=$(( ${SHARD_INDEX:-0} + 1 ))/${NUM_SHARDS:-1}"
"$PYTHON_BIN" "$ROOT/scripts/build_geo_current_family_manifest.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-manifest "$VAL_FAMILY_MANIFEST" \
  --splits $SPLITS \
  --tile-size-px "${TILE_SIZE_PX:-896}" \
  --overlap-px "${OVERLAP_PX:-232}" \
  --keep-margin-px "${KEEP_MARGIN_PX:-116}" \
  --review-crop-pad-px "${REVIEW_CROP_PAD_PX:-64}" \
  --tile-min-mask-ratio "${TILE_MIN_MASK_RATIO:-0.02}" \
  --tile-min-mask-pixels "${TILE_MIN_MASK_PIXELS:-256}" \
  --shard-index "${SHARD_INDEX:-0}" \
  --num-shards "${NUM_SHARDS:-1}" \
  --search-within-review-bbox \
  --fallback-to-all-if-empty

echo "[GeoCurrentV1] val manifest ready -> $VAL_FAMILY_MANIFEST"
