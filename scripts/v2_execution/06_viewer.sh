#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
stage_capture "06" "$@"
action="${1:-status}"; shift || true
stage_begin "06" "Viewer Web · ${action}"
run_arg "$@"
case "${action}" in
  start)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" start-viewer
    stage_step "Viewer start request accepted"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field viewer.state --value ready --timeout 30 \
      --label "Starting Viewer Web" --show-final-status
    stage_step "Viewer service ready"
    bind_host="$("${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]))["server"];print("127.0.0.1" if v["bind_host"] in {"0.0.0.0","::"} else v["bind_host"])' "${RUN_ROOT}/config.json")"
    web_port="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["server"]["web_port"])' "${RUN_ROOT}/config.json")"
    printf 'Viewer ready and waiting for input: http://%s:%s\n' "${bind_host}" "${web_port}"
    stage_step "Viewer URL generated"
    ;;
  status)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    stage_step "Runtime reachable"
    stage_step "Viewer status retrieved"
    stage_step "Status check completed"
    ;;
  stop)
    bash scripts/v2_execution/04_model.sh stop --run "${RUN_ROOT}"
    stage_step "Runtime shutdown completed"
    stage_step "Viewer stopped"
    stage_step "Port release initiated"
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
