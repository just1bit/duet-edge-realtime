#!/usr/bin/env bash
#SBATCH --job-name=edge-v1-original
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${EDGE_ENV_INIT:?export EDGE_ENV_INIT=/absolute/path/init_env.sh}"
: "${DUET_EDGE_ROOT:?export DUET_EDGE_ROOT=/absolute/path/to/duet-edge}"
: "${EDGE_CHECKPOINT:?export EDGE_CHECKPOINT=/absolute/path/train-1800.pt}"
: "${EDGE_OUTPUT_DIR:?export EDGE_OUTPUT_DIR=/absolute/path/realtime-runs}"
source "${EDGE_ENV_INIT}"
conda activate "${EDGE_CONDA_ENV:-edge}"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
OUT="${EDGE_OUTPUT_DIR}/original-model-smoke-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cd "${DUET_EDGE_ROOT}"
python eval/run_ep1800_cfg_sweep.py \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --data_dir "${DUET_EDGE_ROOT}/data" \
  --eval_root "${OUT}/eval" \
  --render_root "${OUT}/renders" \
  --smoke \
  --no_render
test -f "${OUT}/eval/summary.json"
echo "original model smoke passed: ${OUT}/eval/summary.json"
