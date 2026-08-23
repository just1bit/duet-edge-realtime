#!/usr/bin/env bash
set -euo pipefail

V2_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${V2_SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="$(dirname -- "${REALTIME_ROOT}")"
PYTHON_BIN="${PYTHON_BIN:-${REALTIME_ROOT}/.venv/bin/python3}"
STATE_FILE="${REALTIME_ROOT}/outputs/.v2-current"

STAGE_NUMBER=""
STAGE_TITLE=""
STAGE_TOTAL=0
STAGE_CURRENT=0
STAGE_FINISHED=0

progress_bar() {
  local current="$1" total="$2" label="$3" width=28 filled empty percent
  local filled_bar empty_bar
  if (( total <= 0 )); then
    printf '[............................]  --%%  %s\n' "${label}"
    return
  fi
  percent=$((current * 100 / total))
  filled=$((current * width / total))
  empty=$((width - filled))
  printf -v filled_bar '%*s' "${filled}" ''
  printf -v empty_bar '%*s' "${empty}" ''
  filled_bar="${filled_bar// /=}"
  empty_bar="${empty_bar// /.}"
  printf '[%s%s] %3d%%  %s\n' "${filled_bar}" "${empty_bar}" "${percent}" "${label}"
}

stage_begin() {
  STAGE_NUMBER="$1"
  STAGE_TITLE="$2"
  STAGE_TOTAL="$3"
  STAGE_CURRENT=0
  STAGE_FINISHED=0
  printf '\nStage %s · %s\n' "${STAGE_NUMBER}" "${STAGE_TITLE}"
  progress_bar 0 "${STAGE_TOTAL}" "Ready to start"
  trap 'stage_finish $?' EXIT
}

stage_step() {
  STAGE_CURRENT=$((STAGE_CURRENT + 1))
  progress_bar "${STAGE_CURRENT}" "${STAGE_TOTAL}" "$1"
}

stage_finish() {
  local status="$1"
  (( STAGE_FINISHED == 0 )) || return
  STAGE_FINISHED=1
  trap - EXIT
  if (( status == 0 )); then
    STAGE_CURRENT="${STAGE_TOTAL}"
    printf 'Stage %s SUCCESS · %s\n' "${STAGE_NUMBER}" "${STAGE_TITLE}"
  else
    printf 'Stage %s FAILED · %s (exit code %s)\n' \
      "${STAGE_NUMBER}" "${STAGE_TITLE}" "${status}" >&2
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
