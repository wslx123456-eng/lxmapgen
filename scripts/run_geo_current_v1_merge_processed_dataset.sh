#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MERGED_ROOT="${MERGED_ROOT:-$ROOT/outputs/geo_current_v1_merged}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_geo_current_v1_merge_processed_dataset.sh <shard_root_1> [<shard_root_2> ...]" >&2
  exit 1
fi

echo "[GeoCurrentV1] merge processed shards -> $MERGED_ROOT"
"$PYTHON_BIN" "$ROOT/scripts/merge_geo_current_v1_processed_shards.py" \
  --output-root "$MERGED_ROOT" \
  --shard-roots "$@"
