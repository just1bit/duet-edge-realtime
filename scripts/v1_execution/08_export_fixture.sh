#!/usr/bin/env bash
# Export a real model fixture for benchmarking and final validation.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 08 export-fixture "Real fixture export is not applicable to the local profile."
load_run
next_action="Review the model log, apply the indicated action, and repeat this script."
require_file 08 export-fixture "${next_action}" "${RUN_ROOT}/input_motion.pkl"
run_stage 08 export-fixture "${next_action}" \
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/export_fixture.py" \
  --checkpoint "${EDGE_CHECKPOINT}" --duet-edge-root "${DUET_EDGE_ROOT}" \
  --motion "${RUN_ROOT}/input_motion.pkl" --root-scaled false \
  --output "${RUN_ROOT}/real_fixture.npz"
