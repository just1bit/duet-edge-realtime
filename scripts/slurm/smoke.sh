#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${DUET_EDGE_ROOT:?export DUET_EDGE_ROOT=/absolute/path/to/duet-edge}"
: "${EDGE_CHECKPOINT:?export EDGE_CHECKPOINT=/absolute/path/train-1800.pt}"
: "${EDGE_INPUT_MOTION:?export EDGE_INPUT_MOTION=/absolute/path/motion.pkl}"
: "${ROOT_SCALED:?export ROOT_SCALED=true for motions_sliced or false for motions}"
: "${EDGE_OUTPUT_DIR:?export EDGE_OUTPUT_DIR=/absolute/path/realtime-runs}"
RUN_ID="smoke-${SLURM_JOB_ID}"
mkdir -p logs "${EDGE_OUTPUT_DIR}"
export PYTHONPATH="${PWD}/src"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json --input-format aist \
  --root-scaled "${ROOT_SCALED}" --clock virtual --sink ndjson \
  --run-id "${RUN_ID}"
RUN_DIR="${EDGE_OUTPUT_DIR}/${RUN_ID}"
python scripts/check_run.py --summary "${RUN_DIR}/summary.json" --ndjson "${RUN_DIR}/stream.ndjson"
python scripts/export_fixture.py \
  --checkpoint "${EDGE_CHECKPOINT}" --duet-edge-root "${DUET_EDGE_ROOT}" \
  --motion "${EDGE_INPUT_MOTION}" --root-scaled "${ROOT_SCALED}" \
  --output "${RUN_DIR}/real-fixture.npz"
