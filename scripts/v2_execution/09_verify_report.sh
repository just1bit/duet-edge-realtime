#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run=""; long_input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run="$2"; shift 2 ;;
    --long-input) long_input="--long-input"; shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
stage_begin "09" "Verification and Report" 3
load_run "${run}"
stage_step "Run evidence located"
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py report --run "${RUN_ROOT}" ${long_input}
stage_step "Automated acceptance gates completed"
stage_step "Acceptance report generated"
