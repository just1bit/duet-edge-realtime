#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
action="${1:-status}"; shift || true
run_arg "$@"
case "${action}" in
  start)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" start-stream
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field stream.state --value ready --timeout 30
    ;;
  status)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    ;;
  stop)
    bash scripts/v2_execution/04_model.sh stop --run "${RUN_ROOT}"
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
