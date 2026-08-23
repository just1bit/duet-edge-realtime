#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run=""
while [[ $# -gt 0 ]]; do case "$1" in --run) run="$2";shift 2;; *) shift;; esac; done
load_run "${run}"
"${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" start-run
timeout_s="$("${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(max(120,int(v["duration_s"]+600)))' "${RUN_ROOT}/input-manifest.json")"
"${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
  --field session.state --value finished --fail-value failed --timeout "${timeout_s}" --interval 2
test -f "${RUN_ROOT}/summary.json"
test -f "${RUN_ROOT}/stream.ndjson"
printf 'Formal run completed: %s\n' "${RUN_ROOT}"
