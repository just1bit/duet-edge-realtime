#!/usr/bin/env bash
# Display timing values derived from the selected benchmark candidate.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 12 show-recommendation "CUDA configuration recommendations are not applicable to the local profile."
load_run
steps="${1:-50}"
run_stage 12 show-recommendation-${steps} "Review the generated benchmark and select a measured candidate." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/config_recommendation.py" \
  --benchmark "${RUN_ROOT}/evidence/benchmarks/benchmark.json" --steps "${steps}"
