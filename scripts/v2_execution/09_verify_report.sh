#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run=""; release=""
while [[ $# -gt 0 ]]; do case "$1" in --run) run="$2";shift 2;; --release) release="--release";shift;; *) shift;; esac; done
stage_begin "09" "Verification and Report" 3
load_run "${run}"
stage_step "Run evidence located"
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py report --run "${RUN_ROOT}" ${release}
stage_step "Automated acceptance gates completed"
stage_step "Acceptance report generated"
