#!/usr/bin/env bash
# Collect machine, path, hash, port, and browser evidence.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
websocket_port="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("configs/v1.cuda.json"))["server"]["port"])')"
run_stage 03 preflight "Apply the actions shown in the preflight report, then repeat this script." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/preflight.py" \
  --realtime-root "${REALTIME_ROOT}" --duet-edge-root "${DUET_EDGE_ROOT}" \
  --checkpoint "${EDGE_CHECKPOINT}" --motion "${AIST_RAW}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256}" --motion-sha256 "${AIST_RAW_SHA256}" \
  --http-port "${HTTP_PORT}" --websocket-port "${websocket_port}" \
  --output "${RUN_ROOT}/evidence/preflight/preflight.json" --profile "${ACCEPTANCE_PROFILE}"
