#!/usr/bin/env bash
# Compare one candidate stream with the 50-step baseline.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 11 compare-quality "CUDA candidate quality comparison is not applicable to the local profile."
load_run
steps="${1:-}"
if [[ -z "${steps}" ]]; then record_precondition_failure 11 compare-quality "Provide sampling steps and repeat this script." "Usage: bash scripts/v1_execution/11_compare_quality.sh <sampling-steps>"; fi
next_action="Review the quality evidence and select another candidate when useful."
select_latest_service_run 11 compare-quality-${steps} "${next_action}" benchmark-50
baseline_run="${SERVICE_RUN}"
select_latest_service_run 11 compare-quality-${steps} "${next_action}" "benchmark-${steps}"
candidate_run="${SERVICE_RUN}"
run_stage 11 compare-quality-${steps} "${next_action}" \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/compare_quality.py" \
  --fixture "${RUN_ROOT}/real_fixture.npz" \
  --baseline-ndjson "${baseline_run}/stream.ndjson" \
  --candidate-ndjson "${candidate_run}/stream.ndjson" \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --output "${RUN_ROOT}/evidence/benchmarks/quality-${steps}.json"
