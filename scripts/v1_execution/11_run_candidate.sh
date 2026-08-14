#!/usr/bin/env bash
# Run one selected sampling-step candidate benchmark.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 11 run-candidate "CUDA candidate tuning is not applicable to the local profile."
load_run
steps="${1:-}"
if [[ -z "${steps}" ]]; then record_precondition_failure 11 run-candidate "Provide sampling steps and repeat this script." "Usage: bash scripts/v1_execution/11_run_candidate.sh <sampling-steps>"; fi
config="${RUN_ROOT}/candidate-configs/v1.cuda.steps-${steps}.json"
next_action="Review the candidate log, apply the indicated action, and repeat this script."
require_file 11 run-candidate-${steps} "${next_action}" "${config}"
require_file 11 run-candidate-${steps} "${next_action}" "${RUN_ROOT}/real_fixture.npz"
run_id="$(next_service_run_id "benchmark-${steps}")"
run_stage 11 run-candidate-${steps} "${next_action}" \
  "${PYTHON_BIN}" -m duet_edge_realtime.service --config "${config}" \
  --duet-edge-root "${DUET_EDGE_ROOT}" --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/real_fixture.npz" --input-format fixture --output-dir "${RUN_ROOT}" \
  --loop "${BASELINE_LOOPS}" --clock virtual --sink ndjson --run-id "${run_id}"
