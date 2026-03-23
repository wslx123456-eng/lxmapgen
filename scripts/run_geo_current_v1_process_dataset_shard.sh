#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/dataset/zsy/dataset-extracted}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$ROOT/outputs/geo_current_v1_processed}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SPLITS="${SPLITS:-train val}"
SHARD_TAG="${SHARD_TAG:-shard_$(printf '%02d' "$SHARD_INDEX")_of_$(printf '%02d' "$NUM_SHARDS")}"
SHARD_ROOT="${SHARD_ROOT:-$PROCESSED_ROOT/$SHARD_TAG}"
FAMILY_MANIFEST="${FAMILY_MANIFEST:-$SHARD_ROOT/family_manifest.jsonl}"
STAGE_A_DATASET="${STAGE_A_DATASET:-$SHARD_ROOT/stage_a/dataset}"
STAGE_B_DATASET="${STAGE_B_DATASET:-$SHARD_ROOT/stage_b/dataset}"

mkdir -p "$SHARD_ROOT"

echo "[GeoCurrentV1] process shard root=$SHARD_ROOT shard=$((SHARD_INDEX + 1))/$NUM_SHARDS"
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

echo "[GeoCurrentV1] export shard Stage A + Stage B datasets -> $SHARD_ROOT"
"$PYTHON_BIN" "$ROOT/scripts/export_llamafactory_both_from_geo_current_family_manifest.py" \
  --family-manifest "$FAMILY_MANIFEST" \
  --output-root "$SHARD_ROOT" \
  --splits $SPLITS \
  --use-system-prompt \
  --resample-step-px "${RESAMPLE_STEP_PX:-4.0}" \
  --boundary-tol-px "${BOUNDARY_TOL_PX:-2.5}" \
  --trace-points "${TRACE_POINTS:-8}" \
  --state-mixture-mode "${STATE_MIXTURE_MODE:-full}" \
  --state-no-state-ratio "${STATE_NO_STATE_RATIO:-0.30}" \
  --state-weak-ratio "${STATE_WEAK_RATIO:-0.40}" \
  --state-full-ratio "${STATE_FULL_RATIO:-0.30}" \
  --state-weak-trace-points "${STATE_WEAK_TRACE_POINTS:-3}" \
  --state-line-dropout "${STATE_LINE_DROPOUT:-0.40}" \
  --state-point-jitter-px "${STATE_POINT_JITTER_PX:-2.0}" \
  --state-truncate-prob "${STATE_TRUNCATE_PROB:-0.30}"

echo "[GeoCurrentV1] shard processing complete -> $SHARD_ROOT"
