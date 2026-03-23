#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
DATASET_ROOT="${DATASET_ROOT:-/dataset/zsy/dataset-extracted}"
QWEN_ROOT="${QWEN_ROOT:-/dataset/zsy/ckpts/qwen}"
MODEL_PATH="${MODEL_PATH:-$QWEN_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/geo_current_v1}"
FAMILY_MANIFEST="${FAMILY_MANIFEST:-$OUTPUT_ROOT/family_manifest.jsonl}"
STAGE_A_OUTPUT="${STAGE_A_OUTPUT:-$OUTPUT_ROOT/stage_a/checkpoints}"
STAGE_B_ROOT="${STAGE_B_ROOT:-$OUTPUT_ROOT/stage_b}"
STAGE_B_DATASET="${STAGE_B_DATASET:-$STAGE_B_ROOT/dataset}"
STAGE_B_CONFIG="$STAGE_B_ROOT/qwen2_5vl_3b_lora_sft.yaml"
STAGE_B_OUTPUT="${STAGE_B_OUTPUT:-$STAGE_B_ROOT/checkpoints}"
BUILD_MANIFEST_IF_MISSING="${BUILD_MANIFEST_IF_MISSING:-1}"
EXPORT_STAGE_B_DATASET="${EXPORT_STAGE_B_DATASET:-1}"
TRAIN_STAGE_A_IF_MISSING="${TRAIN_STAGE_A_IF_MISSING:-1}"

mkdir -p "$STAGE_B_ROOT"

detect_precision_mode() {
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception:
    print("fp16")
    raise SystemExit(0)
if not torch.cuda.is_available():
    print("fp16")
    raise SystemExit(0)
major, minor = (0, 0)
try:
    version = tuple(int(x) for x in str(torch.__version__).split("+")[0].split(".")[:2])
    major, minor = version
except Exception:
    pass
if major < 2:
    print("fp16")
    raise SystemExit(0)
try:
    if torch.cuda.is_bf16_supported():
        print("bf16")
    else:
        print("fp16")
except Exception:
    print("fp16")
PY
}

if [[ ! -f "$FAMILY_MANIFEST" && "$BUILD_MANIFEST_IF_MISSING" == "1" ]]; then
  bash "$ROOT/scripts/run_geo_current_v1_build_manifest.sh"
fi
if [[ "$EXPORT_STAGE_B_DATASET" == "1" && ! -f "$FAMILY_MANIFEST" ]]; then
  echo "[GeoCurrentV1] missing family manifest: $FAMILY_MANIFEST" >&2
  exit 1
fi
if [[ "$EXPORT_STAGE_B_DATASET" != "1" && ! -f "$STAGE_B_DATASET/train.jsonl" ]]; then
  echo "[GeoCurrentV1] missing preprocessed Stage B dataset: $STAGE_B_DATASET/train.jsonl" >&2
  exit 1
fi
if [[ ! -d "$STAGE_A_OUTPUT" && "$TRAIN_STAGE_A_IF_MISSING" == "1" ]]; then
  bash "$ROOT/scripts/run_geo_current_v1_stagea_train_1gpu.sh"
elif [[ ! -d "$STAGE_A_OUTPUT" ]]; then
  echo "[GeoCurrentV1] missing Stage A checkpoint dir: $STAGE_A_OUTPUT" >&2
  exit 1
fi

PRECISION_MODE="${PRECISION_MODE:-$(detect_precision_mode)}"
if [[ "$PRECISION_MODE" == "bf16" ]]; then
  BF16_FLAG="true"
  FP16_FLAG="false"
else
  BF16_FLAG="false"
  FP16_FLAG="true"
fi

if [[ "$EXPORT_STAGE_B_DATASET" == "1" ]]; then
  echo "[GeoCurrentV1] export Stage B dataset -> $STAGE_B_DATASET"
  "$PYTHON_BIN" "$ROOT/scripts/export_llamafactory_state_sft_from_geo_current_family_manifest.py" \
    --family-manifest "$FAMILY_MANIFEST" \
    --output-root "$STAGE_B_DATASET" \
    --splits train val \
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
else
  echo "[GeoCurrentV1] reuse Stage B dataset -> $STAGE_B_DATASET"
fi

cat > "$STAGE_B_CONFIG" <<EOF
### model
model_name_or_path: $MODEL_PATH
adapter_name_or_path: $STAGE_A_OUTPUT
trust_remote_code: true
image_max_pixels: ${IMAGE_MAX_PIXELS:-802816}
video_max_pixels: 16384

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${LORA_RANK:-16}
lora_alpha: ${LORA_ALPHA:-32}
lora_dropout: ${LORA_DROPOUT:-0.05}
lora_target: ${LORA_TARGET:-all}

### dataset
dataset_dir: $STAGE_B_DATASET
media_dir: $STAGE_B_DATASET
dataset: unimapgen_geo_current_state_train
template: qwen2_vl
cutoff_len: ${CUTOFF_LEN:-8192}
val_size: ${VAL_SIZE:-0.02}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESSING_WORKERS:-8}
dataloader_num_workers: ${DATALOADER_WORKERS:-4}

### output
output_dir: $STAGE_B_OUTPUT
logging_steps: ${LOGGING_STEPS:-10}
save_steps: ${SAVE_STEPS:-200}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE:-1}
gradient_accumulation_steps: ${GRAD_ACC_STEPS:-8}
learning_rate: ${LEARNING_RATE:-5.0e-5}
num_train_epochs: ${NUM_TRAIN_EPOCHS:-2.0}
lr_scheduler_type: cosine
warmup_ratio: ${WARMUP_RATIO:-0.03}
bf16: $BF16_FLAG
fp16: $FP16_FLAG
ddp_timeout: 180000000

### eval
eval_strategy: steps
eval_steps: ${EVAL_STEPS:-200}
EOF

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
echo "[GeoCurrentV1] train Stage B on GPU=$CUDA_VISIBLE_DEVICES precision=$PRECISION_MODE"
"$LLAMAFACTORY_BIN" train "$STAGE_B_CONFIG"
