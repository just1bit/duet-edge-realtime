#!/usr/bin/env bash
# Prepare one acceptance run and export its paths into the current shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Run this script with: source scripts/start_acceptance_run.sh\n' >&2
  exit 1
fi

acceptance_script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export REALTIME_ROOT="$(CDPATH= cd -- "${acceptance_script_dir}/.." && pwd -P)"
acceptance_workspace_root="$(dirname -- "${REALTIME_ROOT}")"
export DUET_EDGE_ROOT="${acceptance_workspace_root}/duet-edge"
export EDGE_CHECKPOINT="${acceptance_workspace_root}/data+checkpoint/train-1800.pt"
export AIST_RAW="${acceptance_workspace_root}/data+checkpoint/aist_plusplus_final/motions/gKR_sBM_cAll_d28_mKR2_ch06.pkl"
export RUN_ROOT="${REALTIME_ROOT}/outputs/acceptance-$(date +%Y%m%d-%H%M%S)"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

acceptance_checkpoint_sha="2c948e74400ba78dbec469880746a78dfbb10ed56597917ba2e406cfeb8f9e15"
acceptance_motion_sha="54968522c7a41457a91d5c916e0f504db27c1afd696ca30bd2bc98dc12294e21"

for acceptance_path in \
  "${DUET_EDGE_ROOT}/EDGE.py" \
  "${EDGE_CHECKPOINT}" \
  "${AIST_RAW}"; do
  if [[ ! -f "${acceptance_path}" ]]; then
    printf 'ERROR: missing %s\n' "${acceptance_path}" >&2
    return 1
  fi
done

if [[ "$(sha256sum "${EDGE_CHECKPOINT}" | cut -d' ' -f1)" != "${acceptance_checkpoint_sha}" ]]; then
  printf 'ERROR: checkpoint SHA256 mismatch\n' >&2
  return 1
fi
if [[ "$(sha256sum "${AIST_RAW}" | cut -d' ' -f1)" != "${acceptance_motion_sha}" ]]; then
  printf 'ERROR: AIST motion SHA256 mismatch\n' >&2
  return 1
fi

mkdir -p "${RUN_ROOT}" || return 1
cd "${REALTIME_ROOT}" || return 1
bash scripts/collect_acceptance_preflight.sh \
  "${RUN_ROOT}/acceptance-preflight-post-env.txt" || return 1
python scripts/prepare_aist_motion.py \
  --input "${AIST_RAW}" \
  --output "${RUN_ROOT}/input_motion.pkl" || return 1

printf '\nP0 PREPARATION: PASSED\n'
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"

unset acceptance_script_dir acceptance_workspace_root
unset acceptance_checkpoint_sha acceptance_motion_sha acceptance_path
