#!/usr/bin/env bash
# Validate the manually edited canonical CUDA configuration.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 12 validate-config "CUDA configuration validation is not applicable to the local profile."
load_run
steps="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("configs/v1.cuda.json"))["model"]["sampling_steps"])')"
quality_args=()
if (( steps < 50 )); then
  quality_args=(--quality "${RUN_ROOT}/evidence/benchmarks/quality-${steps}.json")
fi
run_stage 12 validate-config "Apply the displayed configuration actions, record the manual change, and repeat validation." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/config_recommendation.py" \
  --benchmark "${RUN_ROOT}/evidence/benchmarks/benchmark.json" \
  --config configs/v1.cuda.json "${quality_args[@]}" --validate
