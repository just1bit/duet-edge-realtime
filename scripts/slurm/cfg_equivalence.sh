#!/usr/bin/env bash
#SBATCH --job-name=duet-v1-cfg-eq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${EDGE_ENV_INIT:?export EDGE_ENV_INIT=/absolute/path/init_env.sh}"
: "${DUET_EDGE_ROOT:?export DUET_EDGE_ROOT=/absolute/path/to/duet-edge}"
: "${EDGE_CHECKPOINT:?export EDGE_CHECKPOINT=/absolute/path/train-1800.pt}"
: "${EDGE_OUTPUT_DIR:?export EDGE_OUTPUT_DIR=/absolute/path/realtime-runs}"
source "${EDGE_ENV_INIT}"
conda activate "${EDGE_CONDA_ENV:-edge}"
mkdir -p logs "${EDGE_OUTPUT_DIR}"
export PYTHONPATH="${PWD}/src"
export PYTHONUNBUFFERED=1
python scripts/check_cfg_equivalence.py \
  --checkpoint "${EDGE_CHECKPOINT}" --duet-edge-root "${DUET_EDGE_ROOT}" \
  --output "${EDGE_OUTPUT_DIR}/cfg-equivalence-${SLURM_JOB_ID}.json"
