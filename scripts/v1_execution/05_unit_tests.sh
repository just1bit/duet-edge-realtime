#!/usr/bin/env bash
# Run the standard automated test suite.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
run_stage 05 unit-tests "Apply the test guidance and repeat this script." "${PYTHON_BIN}" -m pytest

