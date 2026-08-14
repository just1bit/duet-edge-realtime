#!/usr/bin/env bash
# Run the deterministic real-model CUDA smoke test.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 02 cuda-smoke "CUDA smoke testing is not applicable to the local profile."
run_stage 02 cuda-smoke "Review the CUDA smoke log, apply the recommended runtime action, and repeat this script." \
  env RUN_CUDA_TESTS=1 DUET_EDGE_CHECKPOINT="${EDGE_CHECKPOINT}" DUET_EDGE_ROOT="${DUET_EDGE_ROOT}" \
  "${PYTHON_BIN}" -m pytest -q tests/test_cuda_smoke.py
