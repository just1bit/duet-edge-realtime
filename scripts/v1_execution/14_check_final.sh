#!/usr/bin/env bash
# Validate the final ten-minute stream and performance evidence.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 14 check-final "Final CUDA stream validation is not applicable to the local profile."
load_run
next_action="Apply the validation actions, repeat the affected stage, and run this check again."
select_latest_service_run 14 check-final "${next_action}" final-10min
service_run="${SERVICE_RUN}"
run_stage 14 check-final "${next_action}" \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/check_run.py" \
  --summary "${service_run}/summary.json" \
  --ndjson "${service_run}/stream.ndjson" \
  --duration-min 10 --require-backend cuda --min-inference-samples 100 --require-performance
