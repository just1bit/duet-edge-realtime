#!/usr/bin/env bash
# Create a new acceptance run and make it the active run.

set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

"${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/init_run.py" \
  --realtime-root "${REALTIME_ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  --state-file "${STATE_FILE}" \
  --profile "${ACCEPTANCE_PROFILE}"
