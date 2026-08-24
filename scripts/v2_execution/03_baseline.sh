#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
stage_capture "03" "$@"
stage_begin "03" "Baseline and Automatic Configuration"
run_arg "$@"
backend="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["backend"])' "${RUN_ROOT}/config.json")"
baseline_root="${RUN_ROOT}/evidence/baseline-runs"
baseline_loops=1
[[ "${backend}" != "cuda" ]] || baseline_loops="${V2_BASELINE_LOOPS:-5}"
input_args=()
if [[ "${backend}" == "cuda" ]]; then
  input_args=(--input "${PROJECT_ROOT}/data+checkpoint/baseline_input/baseline_input.pkl" --input-format aist --root-scaled false)
fi
if [[ ! -f "${baseline_root}/baseline/summary.json" ]]; then
  "${PYTHON_BIN}" -m duet_edge_realtime.service \
    --config "${RUN_ROOT}/config.json" --output-dir "${baseline_root}" \
    --run-id baseline --clock realtime --sink ndjson \
    --loop 1 --progress "${input_args[@]}"
fi
stage_step "Quality baseline completed"
timing_summary="${baseline_root}/baseline/summary.json"
if [[ "${baseline_loops}" -gt 1 ]]; then
  timing_summary="${baseline_root}/timing-baseline/summary.json"
  if [[ ! -f "${timing_summary}" ]]; then
    "${PYTHON_BIN}" -m duet_edge_realtime.service \
      --config "${RUN_ROOT}/config.json" --output-dir "${baseline_root}" \
      --run-id timing-baseline --clock realtime --sink ndjson \
      --loop "${baseline_loops}" --progress "${input_args[@]}"
  fi
fi
stage_step "Timing baseline completed"
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py calibrate --run "${RUN_ROOT}" \
  --summary "${timing_summary}" \
  --quality-summary "${baseline_root}/baseline/summary.json"
stage_step "Configuration calibrated and locked"
