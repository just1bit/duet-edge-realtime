#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
run=""
while [[ $# -gt 0 ]]; do case "$1" in --run) run="$2";shift 2;; *) shift;; esac; done
load_run "${run}"
if [[ -f "${RUN_ROOT}/runtime.pid" ]] && kill -0 "$(runtime_pid)" 2>/dev/null; then
  "${PYTHON_BIN}" scripts/v2_execution/lib/runtime_client.py --run "${RUN_ROOT}" shutdown
  for _ in {1..120}; do
    kill -0 "$(runtime_pid)" 2>/dev/null || break
    sleep 1
  done
fi
config="${RUN_ROOT}/config.json"
checkpoint="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["paths"]["checkpoint"])' "${config}")"
engine="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["paths"]["duet_edge_root"])' "${config}")"
motion="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "${RUN_ROOT}/input-manifest.json")"
root_scaled="$("${PYTHON_BIN}" -c 'import json,sys;print(str(bool(json.load(open(sys.argv[1]))["root_scaled"])).lower())' "${RUN_ROOT}/input-manifest.json")"
"${PYTHON_BIN}" scripts/v1_execution/lib/export_fixture.py \
  --checkpoint "${checkpoint}" --duet-edge-root "${engine}" --motion "${motion}" \
  --root-scaled "${root_scaled}" --output "${RUN_ROOT}/fixtures/fixture.npz" \
  --golden-output "${RUN_ROOT}/fixtures/recorded_fixture.npz" --windows 3 --steps 50
