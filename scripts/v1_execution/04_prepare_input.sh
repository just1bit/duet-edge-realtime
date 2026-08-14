#!/usr/bin/env bash
# Convert the selected raw AIST++ motion for realtime inference.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
load_run
run_stage 04 prepare-input "Review the input guidance, select the matching source motion, and repeat this script." \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/prepare_aist_motion.py" \
  --input "${AIST_RAW}" --output "${RUN_ROOT}/input_motion.pkl" \
  --metadata "${RUN_ROOT}/evidence/input-motion.json"
