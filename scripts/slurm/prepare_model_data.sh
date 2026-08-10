#!/usr/bin/env bash
#SBATCH --job-name=edge-v1-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${EDGE_ENV_INIT:?export EDGE_ENV_INIT=/absolute/path/init_env.sh}"
: "${DUET_EDGE_ROOT:?export DUET_EDGE_ROOT=/absolute/path/to/duet-edge}"
: "${AIST_ROOT:?export AIST_ROOT=/absolute/path/aist_plusplus_final}"
source "${EDGE_ENV_INIT}"
conda activate "${EDGE_CONDA_ENV:-edge}"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
cd "${DUET_EDGE_ROOT}/data"
python create_dataset.py \
  --dataset_folder "${AIST_ROOT}" \
  --duet \
  --extract-jukebox
for SPLIT in train val test; do
  echo "${SPLIT} motions=$(find "${SPLIT}/motions_sliced" -type f -name '*.pkl' | wc -l)"
  echo "${SPLIT} jukebox=$(find "${SPLIT}/jukebox_feats" -type f -name '*.npy' | wc -l)"
done
