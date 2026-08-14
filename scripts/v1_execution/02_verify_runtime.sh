#!/usr/bin/env bash
# Verify the Python, CUDA, GPU, model, and checkpoint runtime.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
run_stage 02 verify-runtime "Run 02_install_runtime.sh when the verifier recommends an environment update, then repeat verification." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/verify_runtime.py" \
  --duet-edge-root "${DUET_EDGE_ROOT}" --checkpoint "${EDGE_CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256}" --profile "${ACCEPTANCE_PROFILE}"
run_stage 02 capture-environment "Repeat runtime verification, then capture the environment evidence again." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/capture_environment.py" \
  --output-dir "${RUN_ROOT}/evidence/environment" --profile "${ACCEPTANCE_PROFILE}"
