#!/usr/bin/env bash
# Create a complete candidate CUDA JSON for selected sampling steps.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 11 prepare-candidate "CUDA candidate tuning is not applicable to the local profile."
load_run
steps="${1:-}"
if [[ -z "${steps}" ]]; then
  record_precondition_failure 11 prepare-candidate "Provide sampling steps and repeat this script." \
    "Usage: bash scripts/v1_execution/11_prepare_candidate.sh <sampling-steps>"
fi
run_stage 11 prepare-candidate-${steps} "Choose sampling steps from 1 through 1000 and repeat this script." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/candidate_config.py" \
  --source configs/v1.cuda.json --steps "${steps}" \
  --output "${RUN_ROOT}/candidate-configs/v1.cuda.steps-${steps}.json"
