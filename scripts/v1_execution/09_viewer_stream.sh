#!/usr/bin/env bash
# Start the fake realtime stream used for interactive Viewer review.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
run_id="$(next_service_run_id viewer)"
run_stage 09 viewer-stream "Save the current Viewer evidence, prepare the service port, and repeat this script." \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config configs/v1.fake.json \
  --output-dir "${RUN_ROOT}" --loop 10 --clock realtime --sink websocket,ndjson \
  --run-id "${run_id}"
