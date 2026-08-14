#!/usr/bin/env bash
# Record GPU utilization, memory, power, and temperature during the final run.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/common.sh"
skip_if_local 13 monitor-gpu "GPU monitoring is not applicable to the local profile."
load_run
target="$("${PYTHON_BIN}" - "${RUN_ROOT}/evidence/resources" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
existing = list(root.glob("gpu-resources*.csv"))
print(root / ("gpu-resources.csv" if not existing else f"gpu-resources-attempt-{len(existing) + 1}.csv"))
PY
)"
run_stage_accept_signals 13 monitor-gpu "Preserve the current resource evidence and repeat monitoring with the final run." "INT,TERM" \
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop="${GPU_SAMPLE_INTERVAL_SECONDS}" --filename="${target}"
