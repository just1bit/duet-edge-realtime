#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
if [[ "${1:-}" == "--resume" ]]; then
  "${PYTHON_BIN}" scripts/v2_execution/lib/run.py init --resume "$2"
else
  "${PYTHON_BIN}" scripts/v2_execution/lib/run.py init "$@"
fi
