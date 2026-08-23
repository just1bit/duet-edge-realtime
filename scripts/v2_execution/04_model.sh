#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
action="${1:-status}"; shift || true
run_arg "$@"
case "${action}" in
  start)
    test -f "${RUN_ROOT}/config.sha256"
    if [[ -f "${RUN_ROOT}/runtime.pid" ]] && kill -0 "$(runtime_pid)" 2>/dev/null; then
      "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
        --field model.state --value ready --fail-value failed --timeout 900
      exit 0
    fi
    nohup "${PYTHON_BIN}" -m duet_edge_realtime.runtime \
      --config "${RUN_ROOT}/config.json" --run-dir "${RUN_ROOT}" \
      >"${RUN_ROOT}/logs/runtime.log" 2>&1 &
    printf '%s\n' "$!" >"${RUN_ROOT}/runtime.pid"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field model.state --value ready --fail-value failed --timeout 900
    ;;
  status)
    kill -0 "$(runtime_pid)"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    ;;
  stop)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" shutdown
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
