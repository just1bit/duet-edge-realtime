#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${CHECKPOINT:?export CHECKPOINT=/absolute/path/train-1800.pt}"
: "${AIST_MOTION:?export AIST_MOTION=/absolute/path/motion.pkl}"
: "${ROOT_SCALED:?export ROOT_SCALED=true for motions_sliced or false for motions}"
RUN_ROOT="${RUN_ROOT:-outputs/slurm-smoke-${SLURM_JOB_ID}}"
mkdir -p logs "${RUN_ROOT}"
export PYTHONPATH="${PWD}/src"
python -m duet_edge_realtime.service \
  --backend cuda --input-format aist --input "${AIST_MOTION}" \
  --root-scaled "${ROOT_SCALED}" --checkpoint "${CHECKPOINT}" \
  --duet-edge-root third_party/duet-edge --clock virtual --sink ndjson \
  --require-clean-engine --output-dir "${RUN_ROOT}"
python scripts/check_run.py --summary "${RUN_ROOT}/summary.json" --ndjson "${RUN_ROOT}/stream.ndjson"
python scripts/export_fixture.py \
  --checkpoint "${CHECKPOINT}" --motion "${AIST_MOTION}" \
  --root-scaled "${ROOT_SCALED}" --output "${RUN_ROOT}/real-fixture.npz"
