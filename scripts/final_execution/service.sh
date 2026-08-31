#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNTIME_SERVICE="${SCRIPT_DIR}/runtime_service.sh"
RUN_TOOL="${SCRIPT_DIR}/lib/run.py"
CAPTURE_TOOL="${SCRIPT_DIR}/lib/capture_stage.py"
STATE_FILE="${REALTIME_ROOT}/outputs/.final-run-current"
PYTHON_BIN="${PYTHON_BIN:-${REALTIME_ROOT}/.venv/bin/python3}"

usage() {
  cat >&2 <<EOF
Usage:
  $0 start [--run RUN_ROOT | --template CONFIG] [--mode file|mediapipe] [--full-check]
  $0 mode file|mediapipe [--run RUN_ROOT]
  $0 stop [--run RUN_ROOT]
  $0 status [--run RUN_ROOT]
  $0 test [INPUT.pkl] [--root-scaled true|false] [--run RUN_ROOT]
EOF
}

active_run() {
  [[ -f "${STATE_FILE}" ]] || return 1
  sed -n '1p' "${STATE_FILE}"
}

quick_check() {
  local run_root="$1" input_mode
  printf '\nFinal service quick check\n'
  "${PYTHON_BIN}" -c 'import duet_edge_realtime, numpy, websockets'
  printf '  - Runtime imports ready\n'
  input_mode="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("input",{}).get("mode","file"))' "${run_root}/config.json")"
  if [[ "${input_mode}" == "file" ]]; then
    "${PYTHON_BIN}" "${RUN_TOOL}" input --run "${run_root}"
    printf '  - Default file input structure ready\n'
  else
    printf '  - MediaPipe input is externally managed; file input check skipped\n'
  fi
  "${PYTHON_BIN}" - "${run_root}/config.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

config = json.load(open(sys.argv[1]))
for name, item in config["assets"].items():
    digest = hashlib.sha256()
    with Path(item["path"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise SystemExit(f"{name} hash changed")
print("  - Configured asset identities ready")
PY
}

full_check_run() {
  local run_root="$1" backend
  printf '\nStage 02 · Runtime Check and Smoke Test\n'
  "${PYTHON_BIN}" -m pytest -q
  printf '  - Automated tests passed\n'
  quick_check "${run_root}"
  for asset_dir in baseline_input smoke_input stitched_long_input; do
    (cd "${REALTIME_ROOT}/../data+checkpoint/${asset_dir}" && shasum -a 256 -c SHA256SUMS)
  done
  printf '  - Test asset hashes verified\n'
  backend="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["backend"])' "${run_root}/config.json")"
  if [[ "${backend}" == "cuda" && ! -f "${run_root}/evidence/smoke-runs/cuda-smoke/summary.json" ]]; then
    "${PYTHON_BIN}" -m duet_edge_realtime.service \
      --config "${run_root}/config.json" \
      --input "${REALTIME_ROOT}/../data+checkpoint/smoke_input/smoke_input.pkl" \
      --input-format aist --root-scaled false --sampling-steps 5 \
      --output-dir "${run_root}/evidence/smoke-runs" --run-id cuda-smoke \
      --clock virtual --sink ndjson --progress
  fi
  printf '  - Backend smoke run completed\n'
  printf 'Stage 02 SUCCESS · Runtime Check and Smoke Test\n'
}

initialize_run() {
  local template="${1:-}" init_command
  init_command=(
    "${PYTHON_BIN}" "${RUN_TOOL}" init
    --state-file "${STATE_FILE}"
  )
  [[ -z "${template}" ]] || init_command+=(--template "${template}")
  "${PYTHON_BIN}" "${CAPTURE_TOOL}" \
    --stage "01" --state-file "${STATE_FILE}" -- \
    "${init_command[@]}"
}

calibrate_run() {
  local run_root="$1" backend baseline_root timing_summary baseline_loops
  backend="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["backend"])' "${run_root}/config.json")"
  baseline_root="${run_root}/evidence/baseline-runs"
  baseline_loops=1
  [[ "${backend}" != "cuda" ]] || baseline_loops="${FINAL_BASELINE_LOOPS:-5}"

  printf '\nFinal baseline and automatic configuration\n'
  if [[ ! -f "${baseline_root}/baseline/summary.json" ]]; then
    if [[ "${backend}" == "cuda" ]]; then
      "${PYTHON_BIN}" -m duet_edge_realtime.service \
        --config "${run_root}/config.json" --output-dir "${baseline_root}" \
        --run-id baseline --clock realtime --sink ndjson --loop 1 --progress \
        --input "${REALTIME_ROOT}/../data+checkpoint/baseline_input/baseline_input.pkl" \
        --input-format aist --root-scaled false
    else
      "${PYTHON_BIN}" -m duet_edge_realtime.service \
        --config "${run_root}/config.json" --output-dir "${baseline_root}" \
        --run-id baseline --clock realtime --sink ndjson --loop 1 --progress
    fi
  fi
  printf '  - Quality baseline completed\n'

  timing_summary="${baseline_root}/baseline/summary.json"
  if [[ "${baseline_loops}" -gt 1 ]]; then
    timing_summary="${baseline_root}/timing-baseline/summary.json"
    if [[ ! -f "${timing_summary}" ]]; then
      "${PYTHON_BIN}" -m duet_edge_realtime.service \
        --config "${run_root}/config.json" --output-dir "${baseline_root}" \
        --run-id timing-baseline --clock realtime --sink ndjson \
        --loop "${baseline_loops}" --progress \
        --input "${REALTIME_ROOT}/../data+checkpoint/baseline_input/baseline_input.pkl" \
        --input-format aist --root-scaled false
    fi
  fi
  printf '  - Timing baseline completed\n'

  "${PYTHON_BIN}" "${RUN_TOOL}" calibrate --run "${run_root}" \
    --summary "${timing_summary}" \
    --quality-summary "${baseline_root}/baseline/summary.json"
  printf '  - Configuration calibrated and locked\n'
}

start_service() {
  local run_root="" template="" requested_mode="" full_check=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run)
        [[ $# -ge 2 ]] || { printf '%s\n' '--run requires a value' >&2; return 2; }
        run_root="$2"
        shift 2
        ;;
      --template)
        [[ $# -ge 2 ]] || { printf '%s\n' '--template requires a value' >&2; return 2; }
        template="$2"
        shift 2
        ;;
      --full-check)
        full_check=1
        shift
        ;;
      --mode)
        [[ $# -ge 2 ]] || { printf '%s\n' '--mode requires a value' >&2; return 2; }
        [[ "$2" == "file" || "$2" == "mediapipe" ]] || {
          printf '%s\n' '--mode must be file or mediapipe' >&2
          return 2
        }
        requested_mode="$2"
        shift 2
        ;;
      *)
        printf 'Unknown start option: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done

  if [[ -n "${run_root}" && -n "${template}" ]]; then
    printf '%s\n' '--run and --template cannot be used together.' >&2
    return 2
  fi

  cd "${REALTIME_ROOT}"
  if [[ -n "${template}" ]]; then
    initialize_run "${template}"
    run_root="$(active_run)"
  elif [[ -z "${run_root}" ]]; then
    candidate="$(active_run 2>/dev/null || true)"
    if [[ -n "${candidate}" && -f "${candidate}/config.json" ]]; then
      run_root="${candidate}"
    else
      initialize_run
      run_root="$(active_run)"
    fi
  fi

  if [[ ! -f "${run_root}/config.json" ]]; then
    printf 'Run has no config.json: %s\n' "${run_root}" >&2
    return 2
  fi

  if (( full_check == 1 )); then
    "${PYTHON_BIN}" "${CAPTURE_TOOL}" \
      --stage "02" --state-file "${STATE_FILE}" --run-root "${run_root}" -- \
      bash "$0" __full_check --run "${run_root}"
  elif [[ ! -f "${run_root}/config.sha256" ]]; then
    quick_check "${run_root}"
  else
    printf 'Prepared run reused; startup self-tests skipped: %s\n' "${run_root}"
  fi

  if [[ ! -f "${run_root}/config.sha256" ]]; then
    calibrate_run "${run_root}"
  fi

  bash "${RUNTIME_SERVICE}" model start --run "${run_root}"
  bash "${RUNTIME_SERVICE}" stream start --run "${run_root}"
  bash "${RUNTIME_SERVICE}" viewer start --run "${run_root}"
  if [[ -z "${requested_mode}" ]]; then
    requested_mode="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("input",{}).get("mode","file"))' "${run_root}/config.json")"
  fi
  bash "${RUNTIME_SERVICE}" mode "${requested_mode}" --run "${run_root}"
  printf '\nFINAL SERVICE READY\nRun directory: %s\n' "${run_root}"
}

command="${1:-}"
case "${command}" in
  __full_check)
    [[ "${2:-}" == "--run" && -n "${3:-}" ]] || exit 2
    full_check_run "${3}"
    ;;
  start)
    shift
    start_service "$@"
    ;;
  stop|status|mode)
    shift
    bash "${RUNTIME_SERVICE}" "${command}" "$@"
    ;;
  test)
    shift
    bash "${RUNTIME_SERVICE}" test "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
