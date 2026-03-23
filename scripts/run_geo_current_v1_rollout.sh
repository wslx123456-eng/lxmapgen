#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_ROOT="${QWEN_ROOT:-/dataset/zsy/ckpts/qwen}"
MODEL_PATH="${MODEL_PATH:-$QWEN_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/geo_current_v1}"
FAMILY_MANIFEST="${FAMILY_MANIFEST:-$OUTPUT_ROOT/family_manifest.jsonl}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$OUTPUT_ROOT/rollout_eval}"
ADAPTER_PATH="${ADAPTER_PATH:-$OUTPUT_ROOT/stage_b/checkpoints}"

if [[ ! -f "$FAMILY_MANIFEST" ]]; then
  bash "$ROOT/scripts/run_geo_current_v1_build_manifest.sh"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
echo "[GeoCurrentV1] rollout -> $ROLLOUT_OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/rollout_predict_qwen2_5vl_from_geo_current_family_manifest.py" \
  --family-manifest "$FAMILY_MANIFEST" \
  --output-root "$ROLLOUT_OUTPUT" \
  --split "${ROLLOUT_SPLIT:-val}" \
  --max-families "${MAX_FAMILIES:-16}" \
  --base-model "$MODEL_PATH" \
  --adapter "$ADAPTER_PATH" \
  --engine "${ENGINE:-custom}" \
  --device "cuda:0" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --resample-step-px "${RESAMPLE_STEP_PX:-4.0}" \
  --boundary-tol-px "${BOUNDARY_TOL_PX:-2.5}" \
  --trace-points "${TRACE_POINTS:-8}" \
  --state-mode "${STATE_MODE:-full}" \
  --use-patch-only-prompt-when-empty \
  --export-visualizations
