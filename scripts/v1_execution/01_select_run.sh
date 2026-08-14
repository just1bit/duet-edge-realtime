#!/usr/bin/env bash
# Select an existing acceptance run for subsequent stage scripts.

set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

selected="${1:-}"
if [[ -z "${selected}" ]]; then
  printf 'Usage: bash scripts/v1_execution/01_select_run.sh <run-directory>\n' >&2
  exit 2
fi
case "${selected}" in
  /*) ;;
  *) selected="${REALTIME_ROOT}/${selected}" ;;
esac
if [[ ! -f "${selected}/run-metadata.json" ]]; then
  printf 'Choose an acceptance run containing run-metadata.json.\n' >&2
  exit 1
fi
printf '%s\n' "$(CDPATH= cd -- "${selected}" && pwd -P)" >"${STATE_FILE}"
printf 'Active acceptance run: %s\n' "$(sed -n '1p' "${STATE_FILE}")"
