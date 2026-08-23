#!/usr/bin/env bash
set -euo pipefail

V2_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${V2_SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="$(dirname -- "${REALTIME_ROOT}")"
PYTHON_BIN="${PYTHON_BIN:-${REALTIME_ROOT}/.venv/bin/python3}"
STATE_FILE="${REALTIME_ROOT}/outputs/.v2-current"

load_run() {
  local requested="${1:-}"
  if [[ -n "${requested}" ]]; then
    RUN_ROOT="$(CDPATH= cd -- "${requested}" && pwd -P)"
  else
    RUN_ROOT="$(sed -n '1p' "${STATE_FILE}")"
  fi
  [[ -f "${RUN_ROOT}/config.json" ]]
  export RUN_ROOT
}

run_arg() {
  local run=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  load_run "${run}"
}

runtime_pid() {
  sed -n '1p' "${RUN_ROOT}/runtime.pid"
}

cd "${REALTIME_ROOT}"
