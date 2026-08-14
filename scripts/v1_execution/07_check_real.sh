#!/usr/bin/env bash
# Validate the real CUDA smoke stream and summary.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 07 check-real "Real CUDA validation is not applicable to the local profile."
load_run
next_action="Apply the validation actions, rerun 07_run_real.sh, and repeat this check."
select_latest_service_run 07 check-real "${next_action}" real-smoke
service_run="${SERVICE_RUN}"
require_file 07 check-real "${next_action}" "${service_run}/summary.json"
require_file 07 check-real "${next_action}" "${service_run}/stream.ndjson"
run_stage 07 check-real "${next_action}" \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/check_run.py" \
  --summary "${service_run}/summary.json" --ndjson "${service_run}/stream.ndjson" \
  --require-backend cuda --min-inference-samples 2
