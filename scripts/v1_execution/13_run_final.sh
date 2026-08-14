#!/usr/bin/env bash
# Run the final ten-minute realtime CUDA acceptance session.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 13 run-final "The CUDA ten-minute final run is not applicable to the local profile."
load_run
next_action="Review the final service log, apply the indicated action, and repeat the final run."
require_file 13 run-final "${next_action}" "${RUN_ROOT}/real_fixture.npz"
run_id="$(next_service_run_id final-10min)"
run_stage 13 run-final "${next_action}" \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/real_fixture.npz" --input-format fixture --output-dir "${RUN_ROOT}" \
  --loop "${FINAL_LOOPS}" --clock realtime --sink websocket,ndjson --run-id "${run_id}"
