#!/usr/bin/env bash
# Summarize the 50-step CUDA baseline and timing recommendation.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 10 summarize-baseline "CUDA performance benchmarking is not applicable to the local profile."
load_run
run_stage 10 summarize-baseline "Collect at least 100 baseline samples, then repeat this summary." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/summarize_benchmark.py" "${RUN_ROOT}" \
  --pattern 'benchmark-*/summary.json' --steps 50 --min-samples 100 \
  --output evidence/benchmarks/benchmark.json
