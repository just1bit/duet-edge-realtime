#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
stage_capture "05" "$@"
action="${1:-status}"; shift || true
stage_begin "05" "Realtime Stream Service · ${action}"
run_arg "$@"
case "${action}" in
  start)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" start-stream
    stage_step "Start request accepted"
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field stream.state --value ready --timeout 30 \
      --label "Preparing realtime stream service" --show-final-status
    stage_step "Realtime stream service ready"
    ;;
  status)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    stage_step "Runtime reachable"
    stage_step "Stream status retrieved"
    ;;
  stop)
    bash scripts/v2_execution/04_model.sh stop --run "${RUN_ROOT}"
    stage_step "Runtime shutdown completed"
    stage_step "Realtime stream stopped"
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
