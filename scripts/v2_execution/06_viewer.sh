#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
action="${1:-status}"; shift || true
run_arg "$@"
case "${action}" in
  start)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" start-viewer
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" wait \
      --field viewer.state --value ready --timeout 30
    bind_host="$("${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]))["server"];print("127.0.0.1" if v["bind_host"] in {"0.0.0.0","::"} else v["bind_host"])' "${RUN_ROOT}/config.json")"
    web_port="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["server"]["web_port"])' "${RUN_ROOT}/config.json")"
    printf 'Viewer ready and waiting for input: http://%s:%s\n' "${bind_host}" "${web_port}"
    ;;
  status)
    "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" status
    ;;
  stop)
    bash scripts/v2_execution/04_model.sh stop --run "${RUN_ROOT}"
    ;;
  *) printf 'Usage: %s start|status|stop [--run RUN_ROOT]\n' "$0" >&2; exit 2 ;;
esac
