#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat >&2 <<EOF
Usage:
  $0 start [--run RUN_ROOT]
  $0 status [--run RUN_ROOT]
  $0 stop [--run RUN_ROOT]
  $0 debug [--run RUN_ROOT] [--max-observations N]
  $0 doctor [--run RUN_ROOT]
EOF
}

select_run() {
  local run=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  load_run "${run}"
}

producer_pid() {
  sed -n '1p' "${RUN_ROOT}/mediapipe.pid"
}

producer_alive() {
  [[ -f "${RUN_ROOT}/mediapipe.pid" ]] || return 1
  local pid command_line
  pid="$(producer_pid)"
  kill -0 "${pid}" 2>/dev/null || return 1
  command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  [[ "${command_line}" == *"duet_edge_realtime.mediapipe_bridge"* \
    && "${command_line}" == *"${RUN_ROOT}/config.json"* ]]
}

start_producer() {
  select_run "$@"
  if producer_alive; then
    printf 'MediaPipe producer already running (PID %s)\n' "$(producer_pid)"
    return
  fi
  local log_path="${RUN_ROOT}/logs/mediapipe.log"
  mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/evidence"
  printf '\n===== MediaPipe producer start %s =====\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${log_path}"
  nohup "${PYTHON_BIN}" -m duet_edge_realtime.mediapipe_bridge \
    --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" \
    >>"${log_path}" 2>&1 &
  printf '%s\n' "$!" >"${RUN_ROOT}/mediapipe.pid"
  sleep 0.5
  if ! producer_alive; then
    printf 'MediaPipe producer failed to start; see %s\n' "${log_path}" >&2
    return 1
  fi
  printf 'MediaPipe producer started independently (PID %s)\n' "$(producer_pid)"
  printf 'Log: %s\n' "${log_path}"
}

status_producer() {
  select_run "$@"
  local process_state="stopped"
  producer_alive && process_state="running"
  "${PYTHON_BIN}" - "${RUN_ROOT}/evidence/mediapipe-status.json" \
    "${process_state}" "${RUN_ROOT}/mediapipe.pid" <<'PY'
import json, sys
from pathlib import Path

status_path, process_state, pid_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
status = json.loads(status_path.read_text()) if status_path.is_file() else {}
status["process"] = process_state
status["pid"] = int(pid_path.read_text()) if pid_path.is_file() else None
print(json.dumps(status, indent=2))
PY
  [[ "${process_state}" == "running" ]]
}

stop_producer() {
  select_run "$@"
  if ! producer_alive; then
    printf 'MediaPipe producer is not running.\n'
    rm -f -- "${RUN_ROOT}/mediapipe.pid"
    return
  fi
  local pid
  pid="$(producer_pid)"
  kill -TERM "${pid}"
  for _ in {1..100}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    printf 'MediaPipe producer did not stop within 10 seconds (PID %s).\n' "${pid}" >&2
    return 1
  fi
  rm -f -- "${RUN_ROOT}/mediapipe.pid"
  printf 'MediaPipe producer stopped. Service remains in its current input mode.\n'
}

debug_producer() {
  local run="" max_observations="0"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run="$2"; shift 2 ;;
      --max-observations) max_observations="$2"; shift 2 ;;
      *) printf 'Unknown debug option: %s\n' "$1" >&2; return 2 ;;
    esac
  done
  load_run "${run}"
  "${PYTHON_BIN}" -m duet_edge_realtime.mediapipe_bridge \
    --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" \
    --max-observations "${max_observations}" --log-level DEBUG
}

doctor_producer() {
  select_run "$@"
  "${PYTHON_BIN}" -m duet_edge_realtime.mediapipe_bridge \
    --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" --doctor
}

command="${1:-}"
shift || true
case "${command}" in
  start) start_producer "$@" ;;
  status) status_producer "$@" ;;
  stop) stop_producer "$@" ;;
  debug) debug_producer "$@" ;;
  doctor) doctor_producer "$@" ;;
  -h|--help) usage ;;
  *) usage; exit 2 ;;
esac
