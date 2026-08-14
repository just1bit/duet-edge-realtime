#!/usr/bin/env bash
# Run the virtual-clock fake backend end to end.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
run_id="$(next_service_run_id p1-fake)"
run_stage 06 run-fake "Review the service log, apply the indicated action, and repeat this script." \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config configs/v1.fake.json \
  --output-dir "${RUN_ROOT}" --clock virtual --sink ndjson --run-id "${run_id}"
