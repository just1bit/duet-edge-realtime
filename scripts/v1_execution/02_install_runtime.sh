#!/usr/bin/env bash
# Install the pinned acceptance runtime into the active Python environment.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
if [[ "${ACCEPTANCE_PROFILE}" == "local" ]]; then
  run_stage 02 install-runtime "Repeat 02_verify_runtime.sh after installation completes." \
    "${PYTHON_BIN}" -m pip install -e ".[dev]"
  exit 0
fi
run_stage 02 install-runtime "Repeat 02_verify_runtime.sh after installation completes." \
  bash -c 'set -e; command -v g++ >/dev/null; "$1" -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128; "$1" -m pip install -r "$2"; "$1" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"; "$1" -m pip install -e ".[dev]"' \
  acceptance-install "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/requirements.txt"
