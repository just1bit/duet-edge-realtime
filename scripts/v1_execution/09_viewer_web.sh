#!/usr/bin/env bash
# Serve the browser Viewer for interactive review.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
printf 'Open http://127.0.0.1:%s and connect to the configured WebSocket endpoint.\n' "${HTTP_PORT}"
run_stage_accept_signals 09 viewer-web "Prepare the HTTP port and repeat this script." "INT,TERM" \
  "${PYTHON_BIN}" -m http.server "${HTTP_PORT}" --directory web
