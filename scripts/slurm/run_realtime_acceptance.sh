#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-accept
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${CHECKPOINT:?export CHECKPOINT=/absolute/path/train-1800.pt}"
: "${FIXTURE:?export FIXTURE=/absolute/path/normalized-fixture.npz}"
: "${SAMPLING_STEPS:?export SAMPLING_STEPS=the accepted benchmark value}"
: "${PLAYOUT_DELAY_S:?export PLAYOUT_DELAY_S=the measured p99 plus >=0.1s}"
RUN_ROOT="${RUN_ROOT:-outputs/slurm-acceptance-${SLURM_JOB_ID}}"
mkdir -p logs "${RUN_ROOT}"
export PYTHONPATH="${PWD}/src"
# Set LOOP_COUNT so the fixture covers at least 10 minutes. The default 120
# is correct for the 150-frame (5s) fixture produced by export_fixture.py.
LOOP_COUNT="${LOOP_COUNT:-120}"
python -m duet_edge_realtime.service \
  --backend cuda --input-format fixture --input "${FIXTURE}" --loop "${LOOP_COUNT}" \
  --checkpoint "${CHECKPOINT}" --duet-edge-root third_party/duet-edge \
  --sampling-steps "${SAMPLING_STEPS}" --playout-delay-s "${PLAYOUT_DELAY_S}" \
  --clock realtime --sink ndjson,websocket --require-clean-engine \
  --output-dir "${RUN_ROOT}"
python scripts/check_run.py \
  --summary "${RUN_ROOT}/summary.json" --ndjson "${RUN_ROOT}/stream.ndjson" \
  --duration-min 10 --require-performance
