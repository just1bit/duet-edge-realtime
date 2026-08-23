#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run=""; release=""
while [[ $# -gt 0 ]]; do case "$1" in --run) run="$2";shift 2;; --release) release="--release";shift;; *) shift;; esac; done
load_run "${run}"
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py report --run "${RUN_ROOT}" ${release}
