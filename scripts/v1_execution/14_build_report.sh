#!/usr/bin/env bash
# Build the final Markdown index from archived automatic evidence.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
run_stage 14 build-report "Review the archived stage results and repeat report generation." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/build_report.py" "${RUN_ROOT}"
# Refresh after the archived result exists so this build action indexes itself.
"${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/build_report.py" "${RUN_ROOT}"
