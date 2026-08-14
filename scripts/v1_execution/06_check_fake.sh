#!/usr/bin/env bash
# Validate the fake backend stream and summary.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
next_action="Apply the validation actions, rerun 06_run_fake.sh, and repeat this check."
select_latest_service_run 06 check-fake "${next_action}" p1-fake
service_run="${SERVICE_RUN}"
require_file 06 check-fake "${next_action}" "${service_run}/summary.json"
require_file 06 check-fake "${next_action}" "${service_run}/stream.ndjson"
run_stage 06 check-fake "${next_action}" \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/check_run.py" \
  --summary "${service_run}/summary.json" --ndjson "${service_run}/stream.ndjson" \
  --require-backend fake
