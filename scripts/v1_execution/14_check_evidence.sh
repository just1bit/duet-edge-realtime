#!/usr/bin/env bash
# Summarize applicable resources and verify automatic evidence.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
resource_status=0
if [[ "${ACCEPTANCE_PROFILE}" == "local" ]]; then
  skip_stage 14 summarize-resources "GPU resource summaries are not applicable to the local profile."
else
  resource_csv="$("${PYTHON_BIN}" - "${RUN_ROOT}/evidence/resources" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = list(root.glob("gpu-resources*.csv"))
print(max(files, key=lambda path: path.stat().st_mtime_ns) if files else "")
PY
)"
  if [[ -z "${resource_csv}" ]]; then
    record_precondition_failure 14 summarize-resources \
      "Record the final run and GPU monitor together, then repeat this action." \
      "Run 13_monitor_gpu.sh with the final session, then continue." || resource_status=$?
  else
    run_stage 14 summarize-resources "Record the final run and GPU monitor together, then repeat this action." \
      "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/resource_summary.py" \
      --input "${resource_csv}" \
      --output "${RUN_ROOT}/evidence/resources/gpu-summary.json" || resource_status=$?
  fi
fi
evidence_status=0
run_stage 14 check-evidence "Complete the listed automatic evidence, then repeat this check." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/check_evidence.py" "${RUN_ROOT}" \
  --profile "${ACCEPTANCE_PROFILE}" || evidence_status=$?
if (( resource_status != 0 )); then exit "${resource_status}"; fi
exit "${evidence_status}"
