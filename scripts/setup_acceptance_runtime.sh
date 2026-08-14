#!/usr/bin/env bash
# Install the pinned RTX 5090 acceptance runtime, then verify it end to end.

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(CDPATH= cd -- "${script_dir}/.." && pwd -P)"

cd "${repo_root}"

printf '\n[1/7] Checking Python 3.10\n'
python -c 'import sys; assert sys.version_info[:2] == (3, 10), f"Python 3.10 required, got {sys.version}"; print(sys.version)'

if ! command -v g++ >/dev/null 2>&1; then
  printf '\nERROR: g++ is required to build PyTorch3D.\n' >&2
  printf 'Ask an administrator to run: sudo apt update && sudo apt install build-essential\n' >&2
  exit 1
fi

printf '\n[2/7] Installing PyTorch 2.7.0 with CUDA 12.8\n'
python -m pip install \
  torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

printf '\n[3/7] Installing Duet-EDGE runtime dependencies\n'
python -m pip install -r scripts/acceptance-runtime-requirements.txt

printf '\n[4/7] Building PyTorch3D for the installed PyTorch\n'
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@stable'

printf '\n[5/7] Installing duet-edge-realtime and test tools\n'
python -m pip install -e '.[dev]'

printf '\n[6/7] Running dependency and CUDA compatibility checks\n'
python scripts/verify_acceptance_runtime.py
python -m pip check

printf '\n[7/7] Running real Duet-EDGE single-window CUDA inference\n'
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
RUN_CUDA_TESTS=1 \
DUET_EDGE_CHECKPOINT="${repo_root}/../data+checkpoint/train-1800.pt" \
DUET_EDGE_ROOT="${repo_root}/../duet-edge" \
python -m pytest -q tests/test_cuda_smoke.py

evidence_dir="${repo_root}/outputs/environment-evidence"
mkdir -p "${evidence_dir}"
python -m pip freeze >"${evidence_dir}/pip-freeze.txt"
python -m torch.utils.collect_env >"${evidence_dir}/torch-environment.txt"

printf '\n============================================================\n'
printf 'COMPATIBILITY GATE: PASSED\n'
printf 'Evidence: %s\n' "${evidence_dir}"
printf 'You may now continue with P0 in the acceptance guide.\n'
printf '============================================================\n'
