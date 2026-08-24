#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
stage_capture "07" "$@"
run=""; input=""; root_scaled=""
while [[ $# -gt 0 ]]; do case "$1" in --run) run="$2";shift 2;; --input) input="$2";shift 2;; --root-scaled) root_scaled="$2";shift 2;; *) shift;; esac; done
stage_begin "07" "Prepare and Lock Formal Input"
load_run "${run}"
args=(input --run "${RUN_ROOT}" --lock)
[[ -z "${input}" ]] || args+=(--input "${input}")
[[ -z "${root_scaled}" ]] || args+=(--root-scaled "${root_scaled}")
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py "${args[@]}"
stage_step "Input structure validated"
stage_step "Input identity and hash recorded"
stage_step "Formal input manifest locked"
