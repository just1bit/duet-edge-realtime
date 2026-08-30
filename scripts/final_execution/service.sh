#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
V2_DIR="${REALTIME_ROOT}/scripts/v2_execution"
V2_SERVICE="${V2_DIR}/service.sh"
STATE_FILE="${REALTIME_ROOT}/outputs/.run-current"
PYTHON_BIN="${PYTHON_BIN:-${REALTIME_ROOT}/.venv/bin/python3}"

usage() {
  cat >&2 <<EOF
Usage:
  $0 start [--run RUN_ROOT | --template CONFIG] [--full-check]
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
  local run_root="$1"
  printf '\nFinal service quick check\n'
  "${PYTHON_BIN}" -c 'import duet_edge_realtime, numpy, websockets'
  printf '  - Runtime imports ready\n'
  "${PYTHON_BIN}" "${V2_DIR}/lib/run.py" input --run "${run_root}"
  printf '  - Default input structure ready\n'
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

calibrate_run() {
  local run_root="$1" backend baseline_root timing_summary baseline_loops
  backend="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["backend"])' "${run_root}/config.json")"
  baseline_root="${run_root}/evidence/baseline-runs"
  baseline_loops=1
  [[ "${backend}" != "cuda" ]] || baseline_loops="${V2_BASELINE_LOOPS:-5}"

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

  "${PYTHON_BIN}" "${V2_DIR}/lib/run.py" calibrate --run "${run_root}" \
    --summary "${timing_summary}" \
    --quality-summary "${baseline_root}/baseline/summary.json"
  printf '  - Configuration calibrated and locked\n'
}

start_service() {
  local run_root="" template="" full_check=0
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
    bash "${V2_DIR}/01_run.sh" --template "${template}"
    run_root="$(active_run)"
  elif [[ -z "${run_root}" ]]; then
    candidate="$(active_run 2>/dev/null || true)"
    if [[ -n "${candidate}" && -f "${candidate}/config.json" ]]; then
      run_root="${candidate}"
    else
      bash "${V2_DIR}/01_run.sh"
      run_root="$(active_run)"
    fi
  fi

  if [[ ! -f "${run_root}/config.json" ]]; then
    printf 'Run has no config.json: %s\n' "${run_root}" >&2
    return 2
  fi

  if (( full_check == 1 )); then
    bash "${V2_DIR}/02_runtime_smoke.sh" --run "${run_root}"
  elif [[ ! -f "${run_root}/config.sha256" ]]; then
    quick_check "${run_root}"
  else
    printf 'Prepared run reused; startup self-tests skipped: %s\n' "${run_root}"
  fi

  if [[ ! -f "${run_root}/config.sha256" ]]; then
    calibrate_run "${run_root}"
  fi

  bash "${V2_SERVICE}" model start --run "${run_root}"
  bash "${V2_SERVICE}" stream start --run "${run_root}"
  bash "${V2_SERVICE}" viewer start --run "${run_root}"
  printf '\nFINAL SERVICE READY\nRun directory: %s\n' "${run_root}"
}

command="${1:-}"
case "${command}" in
  start)
    shift
    start_service "$@"
    ;;
  stop|status)
    shift
    bash "${V2_SERVICE}" "${command}" "$@"
    ;;
  test)
    shift
    bash "${V2_SERVICE}" test "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
