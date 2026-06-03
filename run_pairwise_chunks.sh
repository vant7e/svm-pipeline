#!/bin/bash
#SBATCH --job-name=PAIRWISE_DECODE
#SBATCH --account=YOUR_ACCOUNT

# =====================================
# Resources
# =====================================

#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# =====================================
# Array Jobs
# =====================================

#SBATCH --array=0-263%5

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -euo pipefail

# =====================================
# ENVIRONMENT
# =====================================

module load python/3.10 scipy-stack

source /path/to/your/env/bin/activate

cd /path/to/project

# =====================================
# CONFIG
# =====================================

FEATURE_SPACE="raw"

MANIFEST="manifests/${FEATURE_SPACE}_manifest.csv"

RESULT_DIR="results/pairwise_${FEATURE_SPACE}"

CHUNK_SIZE=200

PAIR_START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
PAIR_END=$((PAIR_START + CHUNK_SIZE))

mkdir -p ${RESULT_DIR}

DONE_FLAG=${RESULT_DIR}/chunk_${PAIR_START}_${PAIR_END}.done

# =====================================
# SKIP COMPLETED
# =====================================

if [ -f "${DONE_FLAG}" ]; then
    echo "====================================="
    echo "Chunk already completed"
    echo "${DONE_FLAG}"
    echo "Skipping..."
    echo "====================================="
    exit 0
fi

# =====================================
# INFO
# =====================================

echo "====================================="
echo "PAIRWISE DECODING"
echo "====================================="

echo "Feature Space: ${FEATURE_SPACE}"

echo "SLURM JOB ID: ${SLURM_JOB_ID}"
echo "ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"

echo "PAIR START: ${PAIR_START}"
echo "PAIR END: ${PAIR_END}"

echo "RESULT DIR: ${RESULT_DIR}"

echo "====================================="

# =====================================
# RUN DECODING
# =====================================

python -m svm.scripts.run_pairwise_decoding \
    --manifest ${MANIFEST} \
    --out_dir ${RESULT_DIR} \
    --pair_column image \
    --pair_mode all \
    --pair_start ${PAIR_START} \
    --pair_end ${PAIR_END} \
    --time_mode full \
    --tmin 0.0 \
    --tmax 0.6 \
    --spatial_preset occipito_temporal \
    --classifier linear_svm \
    --scaler standard \
    --C 1.0 \
    --max_iter 10000 \
    --verbose

# =====================================
# MARK COMPLETE
# =====================================

touch ${DONE_FLAG}

echo "====================================="
echo "DONE"
echo "${DONE_FLAG}"
echo "====================================="
