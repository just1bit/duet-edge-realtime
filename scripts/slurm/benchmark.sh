#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-bench
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
mkdir -p logs "${EDGE_OUTPUT_DIR}"
export PYTHONPATH="${PWD}/src"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Baseline first. Add non-default values only after committing a compatible
# optimization in duet-edge and updating compat/duet-edge.lock.json.
for STEPS in ${BENCHMARK_STEPS:-50}; do
  RUN_ID="benchmark-${SLURM_JOB_ID}-steps-${STEPS}"
  python -m duet_edge_realtime.service \
    --config configs/v1.cuda.json --input-format fixture --input "${FIXTURE}" \
    --loop 16 --sampling-steps "${STEPS}" --clock virtual --sink ndjson \
    --run-id "${RUN_ID}"
done
python scripts/summarize_benchmark.py "${EDGE_OUTPUT_DIR}" \
  --pattern "benchmark-${SLURM_JOB_ID}-steps-*/summary.json" \
  --output "benchmark-${SLURM_JOB_ID}.json"
