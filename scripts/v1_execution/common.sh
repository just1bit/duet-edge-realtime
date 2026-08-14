#!/usr/bin/env bash
# Shared path, state, and evidence helpers for acceptance stage scripts.

set -euo pipefail

ACCEPTANCE_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REALTIME_ROOT="$(CDPATH= cd -- "${ACCEPTANCE_SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="$(dirname -- "${REALTIME_ROOT}")"

# shellcheck source=acceptance.conf
source "${ACCEPTANCE_SCRIPT_DIR}/acceptance.conf"

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
  esac
}

REALTIME_ROOT="$(resolve_project_path "${REALTIME_DIR}")"
DUET_EDGE_ROOT="$(resolve_project_path "${DUET_EDGE_DIR}")"
EDGE_CHECKPOINT="$(resolve_project_path "${CHECKPOINT_PATH}")"
AIST_RAW="$(resolve_project_path "${AIST_RAW_PATH}")"
STATE_FILE="${REALTIME_ROOT}/outputs/.acceptance-current"

export PROJECT_ROOT REALTIME_ROOT DUET_EDGE_ROOT EDGE_CHECKPOINT AIST_RAW
if [[ "${ACCEPTANCE_PROFILE}" != "gpu" && "${ACCEPTANCE_PROFILE}" != "local" ]]; then
  printf 'ACCEPTANCE_PROFILE must be gpu or local; received %s.\n' "${ACCEPTANCE_PROFILE}" >&2
  exit 2
fi

export CHECKPOINT_SHA256 AIST_RAW_SHA256 PYTHON_BIN HTTP_PORT ACCEPTANCE_PROFILE
export GPU_SAMPLE_INTERVAL_SECONDS BASELINE_LOOPS FINAL_LOOPS STATE_FILE
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

load_run() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    printf 'Create or select an acceptance run with a Stage 01 script.\n' >&2
    return 1
  fi
  RUN_ROOT="$(sed -n '1p' "${STATE_FILE}")"
  if [[ ! -d "${RUN_ROOT}" ]]; then
    printf 'Select an available acceptance run with 01_select_run.sh.\n' >&2
    return 1
  fi
  if [[ -f "${RUN_ROOT}/run-metadata.json" ]]; then
    local recorded_profile
    recorded_profile="$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("acceptance_profile", "gpu"))' "${RUN_ROOT}/run-metadata.json")"
    if [[ "${recorded_profile}" != "${ACCEPTANCE_PROFILE}" ]]; then
      printf 'Use ACCEPTANCE_PROFILE=%s for the selected run; current profile is %s.\n' \
        "${recorded_profile}" "${ACCEPTANCE_PROFILE}" >&2
      return 1
    fi
  fi
  export RUN_ROOT
}

run_stage() {
  local stage="$1"
  local name="$2"
  local next_action="$3"
  shift 3
  load_run
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/archive_stage.py" \
    --run-root "${RUN_ROOT}" \
    --stage "${stage}" \
    --name "${name}" \
    --next-action "${next_action}" \
    -- "$@"
}

run_stage_accept_signals() {
  local stage="$1"
  local name="$2"
  local next_action="$3"
  local signals="$4"
  shift 4
  load_run
  local signal_args=()
  local item
  IFS=',' read -r -a accepted <<<"${signals}"
  for item in "${accepted[@]}"; do
    signal_args+=(--accept-signal "${item}")
  done
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/archive_stage.py" \
    --run-root "${RUN_ROOT}" --stage "${stage}" --name "${name}" \
    --next-action "${next_action}" "${signal_args[@]}" -- "$@"
}

skip_stage() {
  local stage="$1"
  local name="$2"
  local reason="$3"
  load_run
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/archive_stage.py" \
    --run-root "${RUN_ROOT}" --stage "${stage}" --name "${name}" \
    --next-action "No action is required for this profile." --skip-reason "${reason}"
}

skip_if_local() {
  if [[ "${ACCEPTANCE_PROFILE}" == "local" ]]; then
    skip_stage "$1" "$2" "$3"
    exit 0
  fi
}

record_precondition_failure() {
  local stage="$1"
  local name="$2"
  local next_action="$3"
  local message="$4"
  load_run
  "${PYTHON_BIN}" "${ACCEPTANCE_SCRIPT_DIR}/lib/archive_stage.py" \
    --run-root "${RUN_ROOT}" --stage "${stage}" --name "${name}" \
    --next-action "${next_action}" --precondition-error "${message}"
}

require_file() {
  local stage="$1"
  local name="$2"
  local next_action="$3"
  local path="$4"
  if [[ ! -f "${path}" ]]; then
    record_precondition_failure "${stage}" "${name}" "${next_action}" \
      "Prepare the required file, then repeat this stage: ${path}"
  fi
}

require_directory() {
  local stage="$1"
  local name="$2"
  local next_action="$3"
  local path="$4"
  if [[ ! -d "${path}" ]]; then
    record_precondition_failure "${stage}" "${name}" "${next_action}" \
      "Prepare the required directory, then repeat this stage: ${path}"
  fi
}

next_service_run_id() {
  load_run
  "${PYTHON_BIN}" - "$RUN_ROOT" "$1" <<'PY'
import sys
from pathlib import Path
root, base = Path(sys.argv[1]), sys.argv[2]
existing = [
    path for path in root.iterdir()
    if path.is_dir() and (path.name == base or path.name.startswith(f"{base}-attempt-"))
]
print(base if not existing else f"{base}-attempt-{len(existing) + 1}")
PY
}

select_latest_service_run() {
  load_run
  local stage="$1"
  local name="$2"
  local next_action="$3"
  local base="$4"
  if ! SERVICE_RUN="$("${PYTHON_BIN}" - "$RUN_ROOT" "${base}" <<'PY'
import sys
from pathlib import Path
root, base = Path(sys.argv[1]), sys.argv[2]
existing = [
    path for path in root.iterdir()
    if path.is_dir() and (path.name == base or path.name.startswith(f"{base}-attempt-"))
]
if not existing:
    raise SystemExit(f"Run the service action for {base}, then continue with this stage.")
print(max(existing, key=lambda path: path.stat().st_mtime_ns))
PY
)"; then
    record_precondition_failure "${stage}" "${name}" "${next_action}" \
      "Run the service action for ${base}, then continue with this stage."
  fi
  export SERVICE_RUN
}

cd "${REALTIME_ROOT}"
