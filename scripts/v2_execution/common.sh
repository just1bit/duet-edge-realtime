#!/usr/bin/env bash
set -euo pipefail

V2_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${V2_SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="$(dirname -- "${REALTIME_ROOT}")"
PYTHON_BIN="${PYTHON_BIN:-${REALTIME_ROOT}/.venv/bin/python3}"
STATE_FILE="${REALTIME_ROOT}/outputs/.run-current"

STAGE_NUMBER=""
STAGE_TITLE=""
STAGE_FINISHED=0

stage_capture() {
  local number="$1" run="" status
  shift
  local original_args=("$@")
  [[ "${STAGE_CAPTURE_ACTIVE:-0}" != "1" ]] || return 0
  local capture_args=(--stage "${number}" --state-file "${STATE_FILE}")
  if [[ "${number}" != "01" ]]; then
    while [[ $# -gt 0 ]]; do
      case "$1" in --run) run="$2"; shift 2 ;; *) shift ;; esac
    done
    load_run "${run}"
    capture_args+=(--run-root "${RUN_ROOT}")
  fi
  set +e
  STAGE_CAPTURE_ACTIVE=1 "${PYTHON_BIN}" \
    "${V2_SCRIPT_DIR}/lib/capture_stage.py" "${capture_args[@]}" -- \
    bash "$0" "${original_args[@]}"
  status="$?"
  set -e
  exit "${status}"
}

stage_begin() {
  STAGE_NUMBER="$1"
  STAGE_TITLE="$2"
  STAGE_FINISHED=0
  printf '\nStage %s · %s\n' "${STAGE_NUMBER}" "${STAGE_TITLE}"
  trap 'stage_finish $?' EXIT
}

stage_step() {
  printf '  - %s\n' "$1"
}

stage_finish() {
  local status="$1"
  local message
  (( STAGE_FINISHED == 0 )) || return
  STAGE_FINISHED=1
  trap - EXIT
  if (( status == 0 )); then
    message="Stage ${STAGE_NUMBER} SUCCESS · ${STAGE_TITLE}"
    printf '%s\n' "${message}"
  else
    message="Stage ${STAGE_NUMBER} FAILED · ${STAGE_TITLE} (exit code ${status})"
    printf '%s\n' "${message}" >&2
  fi
}

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
