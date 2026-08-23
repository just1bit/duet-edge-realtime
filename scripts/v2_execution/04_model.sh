#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
action="${1:-status}"; shift || true
stage_begin "04" "Model Service · ${action}" 2
run_arg "$@"
case "${action}" in
  start)
    test -f "${RUN_ROOT}/config.sha256"
    if [[ -f "${RUN_ROOT}/runtime.pid" ]] && kill -0 "$(runtime_pid)" 2>/dev/null; then
      "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
        --field model.state --value ready --fail-value failed --timeout 900 --label "Loading and warming up model"
      stage_step "Existing model process reused"
      stage_step "Model service ready"
      exit 0
    fi
    nohup "${PYTHON_BIN}" -m duet_edge_realtime.runtime \
      --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" \
      >"${RUN_ROOT}/logs/runtime.log" 2>&1 &
    printf '%s\n' "$!" >"${RUN_ROOT}/runtime.pid"
    stage_step "Runtime process started"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field model.state --value ready --fail-value failed --timeout 900 --label "Loading and warming up model"
    stage_step "Model service ready"
    ;;
  status)
    kill -0 "$(runtime_pid)"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    stage_step "Model process is alive"
    stage_step "Status retrieved"
    ;;
  stop)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" shutdown
    stage_step "Shutdown request sent"
    stage_step "Runtime shutdown initiated"
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
