#!/usr/bin/env bash
# Run the 50-step CUDA performance baseline.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 10 run-baseline "CUDA performance benchmarking is not applicable to the local profile."
load_run
next_action="Review the benchmark log, apply the indicated action, and repeat this script."
require_file 10 run-baseline "${next_action}" "${RUN_ROOT}/real_fixture.npz"
steps="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("configs/v1.cuda.json"))["model"]["sampling_steps"])')"
if [[ "${steps}" != "50" ]]; then
  record_precondition_failure 10 run-baseline "${next_action}" \
    "Set the canonical CUDA configuration to the 50-step baseline, then repeat Stage 10."
fi
run_id="$(next_service_run_id benchmark-50)"
run_stage 10 run-baseline "${next_action}" \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/real_fixture.npz" --input-format fixture --output-dir "${RUN_ROOT}" \
  --loop "${BASELINE_LOOPS}" --clock virtual --sink ndjson --run-id "${run_id}"
