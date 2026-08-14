#!/usr/bin/env bash
# Summarize the baseline and all completed candidate benchmarks.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 11 summarize-candidates "CUDA candidate tuning is not applicable to the local profile."
load_run
run_stage 11 summarize-candidates "Collect at least 100 samples for each selected candidate, then repeat this summary." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/summarize_benchmark.py" "${RUN_ROOT}" \
  --pattern 'benchmark-*/summary.json' --min-samples 100 \
  --output evidence/benchmarks/benchmark.json
