#!/usr/bin/env bash
# Run the local WebSocket integration test explicitly.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
run_stage 05 network-tests "Prepare the local loopback port and repeat this script." \
  env RUN_NETWORK_TESTS=1 "${PYTHON_BIN}" -m pytest tests/test_websocket_integration.py
