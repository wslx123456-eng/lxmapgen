#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/scripts/run_geo_current_v1_build_manifest.sh"
bash "$ROOT/scripts/run_geo_current_v1_stagea_train_1gpu.sh"
bash "$ROOT/scripts/run_geo_current_v1_stageb_train_1gpu.sh"
bash "$ROOT/scripts/run_geo_current_v1_rollout.sh"

