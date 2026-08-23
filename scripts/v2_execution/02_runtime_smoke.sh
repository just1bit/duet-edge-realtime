#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run_arg "$@"
"${PYTHON_BIN}" -m pytest
"${PYTHON_BIN}" scripts/v2_execution/lib/run.py input --run "${RUN_ROOT}"
for asset_dir in baseline_input smoke_input stitched_long_input; do
  (cd "${PROJECT_ROOT}/data+checkpoint/${asset_dir}" && shasum -a 256 -c SHA256SUMS)
done
backend="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["backend"])' "${RUN_ROOT}/config.json")"
if [[ "${backend}" == "cuda" && ! -f "${RUN_ROOT}/evidence/smoke-runs/cuda-smoke/summary.json" ]]; then
  "${PYTHON_BIN}" -m duet_edge_realtime.service \
    --config "${RUN_ROOT}/config.json" \
    --input "${PROJECT_ROOT}/data+checkpoint/smoke_input/smoke_input.pkl" \
    --input-format aist --root-scaled false --sampling-steps 5 \
    --output-dir "${RUN_ROOT}/evidence/smoke-runs" --run-id cuda-smoke \
    --clock virtual --sink ndjson
fi
"${PYTHON_BIN}" - "${RUN_ROOT}/config.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
config=json.load(open(sys.argv[1]))
for name, item in config["assets"].items():
    digest=hashlib.sha256()
    with Path(item["path"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest()!=item["sha256"]:
        raise SystemExit(f"{name} hash changed")
print("Runtime, tests, input, and asset hashes are ready.")
PY
