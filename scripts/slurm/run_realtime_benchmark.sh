#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${CHECKPOINT:?export CHECKPOINT=/absolute/path/train-1800.pt}"
: "${FIXTURE:?export FIXTURE=/absolute/path/normalized-fixture.npz}"
RUN_ROOT="${RUN_ROOT:-outputs/slurm-benchmark-${SLURM_JOB_ID}}"
mkdir -p logs "${RUN_ROOT}"
export PYTHONPATH="${PWD}/src"
for STEPS in 50 25 20 10; do
  python -m duet_edge_realtime.service \
    --backend cuda --input-format fixture --input "${FIXTURE}" --loop 16 \
    --checkpoint "${CHECKPOINT}" --duet-edge-root third_party/duet-edge \
    --sampling-steps "${STEPS}" --clock virtual --sink ndjson \
    --require-clean-engine --output-dir "${RUN_ROOT}/steps-${STEPS}"
done
python scripts/summarize_benchmark.py "${RUN_ROOT}"
