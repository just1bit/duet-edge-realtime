#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat >&2 <<EOF
Usage:
  $0 model start [--run RUN_ROOT]
  $0 stream start [--run RUN_ROOT]
  $0 viewer start [--run RUN_ROOT]
  $0 test [INPUT] [--root-scaled true|false] [--run RUN_ROOT]
  $0 status [--run RUN_ROOT]
  $0 stop [--run RUN_ROOT]
EOF
}

selected_run() {
  local run=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run)
        [[ $# -ge 2 ]] || { printf '%s\n' '--run requires a value' >&2; return 2; }
        run="$2"
        shift 2
        ;;
      *) shift ;;
    esac
  done
  printf '%s\n' "${run}"
}

capture_stage() {
  local number="$1" run
  shift
  run="$(selected_run "$@")"
  load_run "${run}"
  if [[ "${STAGE_CAPTURE_ACTIVE:-0}" == "1" ]]; then
    bash "${SCRIPT_DIR}/runtime_service.sh" __stage "${number}" "$@"
    return
  fi
  STAGE_CAPTURE_ACTIVE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/lib/capture_stage.py" \
    --stage "${number}" --state-file "${STATE_FILE}" --run-root "${RUN_ROOT}" -- \
    bash "${SCRIPT_DIR}/runtime_service.sh" __stage "${number}" "$@"
}

runtime_client() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/lib/runtime_client.py" --run "${RUN_ROOT}" "$@"
}

model_stage() {
  local action="${1:-status}"
  shift || true
  stage_begin "04" "Model Service · ${action}"
  run_arg "$@"
  case "${action}" in
    start)
      test -f "${RUN_ROOT}/config.sha256"
      if [[ -f "${RUN_ROOT}/runtime.pid" ]] && kill -0 "$(runtime_pid)" 2>/dev/null; then
        runtime_client wait \
          --field model.state --value ready --fail-value failed --timeout 900 --interval 0.2 \
          --label "Loading and warming up model" --show-final-status
        stage_step "Existing model process reused"
        stage_step "Model service ready"
        return
      fi
      local runtime_log="${RUN_ROOT}/logs/runtime.log"
      if ! printf '\n===== Runtime start %s · Stage 04 PID %s =====\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" >>"${runtime_log}"; then
        printf 'Warning: runtime log unavailable; continuing without archival.\n' >&2
        runtime_log="/dev/null"
      fi
      nohup "${PYTHON_BIN}" -m duet_edge_realtime.runtime \
        --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" \
        >>"${runtime_log}" 2>&1 &
      printf '%s\n' "$!" >"${RUN_ROOT}/runtime.pid"
      stage_step "Runtime process started"
      runtime_client wait \
        --field model.state --value ready --fail-value failed --timeout 900 --interval 0.2 \
        --label "Loading and warming up model" --show-final-status
      stage_step "Model service ready"
      ;;
    status)
      kill -0 "$(runtime_pid)"
      runtime_client status
      stage_step "Model process is alive"
      stage_step "Status retrieved"
      ;;
    stop)
      runtime_client shutdown
      stage_step "Shutdown request sent"
      stage_step "Runtime shutdown initiated"
      ;;
    *) usage; return 2 ;;
  esac
}

stream_stage() {
  stage_begin "05" "Realtime Stream Service · start"
  run_arg "$@"
  runtime_client start-stream
  stage_step "Start request accepted"
  runtime_client wait \
    --field stream.state --value ready --timeout 30 \
    --label "Preparing realtime stream service" --show-final-status
  stage_step "Realtime stream service ready"
}

viewer_stage() {
  stage_begin "06" "Viewer Web · start"
  run_arg "$@"
  runtime_client start-viewer
  stage_step "Viewer start request accepted"
  runtime_client wait \
    --field viewer.state --value ready --timeout 30 \
    --label "Starting Viewer Web" --show-final-status
  stage_step "Viewer service ready"
  local bind_host web_port
  bind_host="$("${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]))["server"];print("127.0.0.1" if v["bind_host"] in {"0.0.0.0","::"} else v["bind_host"])' "${RUN_ROOT}/config.json")"
  web_port="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["server"]["web_port"])' "${RUN_ROOT}/config.json")"
  printf 'Viewer ready and waiting for input: http://%s:%s\n' "${bind_host}" "${web_port}"
  stage_step "Viewer URL generated"
}

assert_test_ready() {
  local status_json
  status_json="$(runtime_client status)"
  printf '%s\n' "${status_json}" | "${PYTHON_BIN}" -c '
import json, sys
status = json.load(sys.stdin)
required = ("model", "stream", "viewer")
not_ready = [name for name in required if status.get(name, {}).get("state") != "ready"]
if not_ready:
    raise SystemExit("Services not ready: " + ", ".join(not_ready))
session_state = status.get("session", {}).get("state")
if session_state in {"preparing", "starting", "running"}:
    raise SystemExit("A formal test is already in progress: " + session_state)
'
}

clear_formal_outputs() {
  rm -f -- \
    "${RUN_ROOT}/effective_config.json" \
    "${RUN_ROOT}/summary.json" \
    "${RUN_ROOT}/stream.ndjson" \
    "${RUN_ROOT}/gate-results.json" \
    "${RUN_ROOT}/report.md" \
    "${RUN_ROOT}/fixtures/fixture.npz" \
    "${RUN_ROOT}/fixtures/recorded_fixture.npz"
}

input_stage() {
  local run="" input="" root_scaled=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run="$2"; shift 2 ;;
      --input) input="$2"; shift 2 ;;
      --root-scaled) root_scaled="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  stage_begin "07" "Prepare and Lock Formal Input"
  load_run "${run}"
  assert_test_ready
  local args=(input --run "${RUN_ROOT}" --lock)
  [[ -z "${input}" ]] || args+=(--input "${input}")
  [[ -z "${root_scaled}" ]] || args+=(--root-scaled "${root_scaled}")
  "${PYTHON_BIN}" "${SCRIPT_DIR}/lib/run.py" "${args[@]}"
  clear_formal_outputs
  stage_step "Input structure validated"
  stage_step "Input identity and hash recorded"
  stage_step "Formal input manifest locked"
}

run_stage() {
  local run=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  stage_begin "08" "Input and Formal Realtime Run"
  load_run "${run}"
  runtime_client start-run
  stage_step "Formal run request accepted"
  local timeout_s
  timeout_s="$("${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]));print(max(120,int(v["duration_s"]+600)))' "${RUN_ROOT}/input-manifest.json")"
  stage_step "Input manifest and run parameters loaded"
  runtime_client wait \
    --field session.state --value finished --fail-value failed --timeout "${timeout_s}" --interval 0.2 \
    --label "Realtime inference and playout"
  stage_step "All input frames processed"
  test -f "${RUN_ROOT}/summary.json"
  test -f "${RUN_ROOT}/stream.ndjson"
  printf 'Formal run completed: %s\n' "${RUN_ROOT}"
  stage_step "Run evidence written"
}

if [[ "${1:-}" == "__stage" ]]; then
  stage="${2:-}"
  shift 2 || true
  case "${stage}" in
    04) model_stage "$@" ;;
    05) stream_stage "$@" ;;
    06) viewer_stage "$@" ;;
    07) input_stage "$@" ;;
    08) run_stage "$@" ;;
    *) usage; exit 2 ;;
  esac
  exit
fi

command="${1:-}"
case "${command}" in
  model|stream|viewer)
    [[ "${2:-}" == "start" ]] || { usage; exit 2; }
    shift 2
    case "${command}" in
      model) capture_stage "04" start "$@" ;;
      stream) capture_stage "05" "$@" ;;
      viewer) capture_stage "06" "$@" ;;
    esac
    ;;
  test)
    shift
    if [[ $# -gt 0 && -n "${1:-}" && "${1:-}" != --* ]]; then
      input="$1"
      shift
      capture_stage "07" --input "${input}" "$@"
    else
      capture_stage "07" "$@"
    fi
    capture_stage "08" "$@"
    ;;
  status|stop)
    shift
    capture_stage "04" "${command}" "$@"
    ;;
  *) usage; exit 2 ;;
esac
