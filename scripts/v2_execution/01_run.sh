#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
stage_begin "01" "Initialize / Resume Run" 2
if [[ "${1:-}" == "--resume" ]]; then
  "${PYTHON_BIN}" scripts/v2_execution/lib/run.py init --resume "$2"
else
  "${PYTHON_BIN}" scripts/v2_execution/lib/run.py init "$@"
fi
stage_step "Run directory prepared"
stage_step "Active run selected"
