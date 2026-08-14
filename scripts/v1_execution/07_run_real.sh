#!/usr/bin/env bash
# Run the real CUDA backend with the prepared AIST++ motion.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 07 run-real "Real CUDA streaming is not applicable to the local profile."
load_run
next_action="Review the CUDA service log, apply the indicated action, and repeat this script."
require_file 07 run-real "${next_action}" "${RUN_ROOT}/input_motion.pkl"
run_id="$(next_service_run_id real-smoke)"
run_stage 07 run-real "${next_action}" \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/input_motion.pkl" --input-format aist --root-scaled false \
  --output-dir "${RUN_ROOT}" --clock virtual --sink ndjson --run-id "${run_id}"
