#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-accept
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${DUET_EDGE_ROOT:?export DUET_EDGE_ROOT=/absolute/path/to/duet-edge}"
: "${EDGE_CHECKPOINT:?export EDGE_CHECKPOINT=/absolute/path/train-1800.pt}"
: "${FIXTURE:?export FIXTURE=/absolute/path/normalized-fixture.npz}"
: "${EDGE_OUTPUT_DIR:?export EDGE_OUTPUT_DIR=/absolute/path/realtime-runs}"
: "${SAMPLING_STEPS:?export SAMPLING_STEPS=the accepted benchmark value}"
: "${PLAYOUT_DELAY_S:?export PLAYOUT_DELAY_S=measured p99 plus at least 0.1s}"
RUN_ID="acceptance-${SLURM_JOB_ID}"
mkdir -p logs "${EDGE_OUTPUT_DIR}"
export PYTHONPATH="${PWD}/src"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
LOOP_COUNT="${LOOP_COUNT:-120}"
python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json --input-format fixture --input "${FIXTURE}" \
  --loop "${LOOP_COUNT}" --sampling-steps "${SAMPLING_STEPS}" \
  --playout-delay-s "${PLAYOUT_DELAY_S}" --clock realtime \
  --sink ndjson,websocket --run-id "${RUN_ID}"
RUN_DIR="${EDGE_OUTPUT_DIR}/${RUN_ID}"
python scripts/check_run.py \
  --summary "${RUN_DIR}/summary.json" --ndjson "${RUN_DIR}/stream.ndjson" \
  --duration-min 10 --require-performance
