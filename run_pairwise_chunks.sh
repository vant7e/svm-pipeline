#!/bin/bash
#SBATCH --job-name=PAIRWISE_RAW
#SBATCH --account=def-jmeltzer

# =====================================
# Resources
# =====================================

#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# =====================================
# Array jobs
#
# total pairs:
#   325 choose 2 = 52650
#
# chunk size:
#   200 pairs / task
#
# total chunks:
#   ceil(52650 / 200) = 264
#
# max concurrent:
#   5 jobs at once
# =====================================

#SBATCH --array=0-263%5

#SBATCH --output=/home/vant7ee/scratch/logs/pairwise_raw/%A_%a.out
#SBATCH --error=/home/vant7ee/scratch/logs/pairwise_raw/%A_%a.err

set -euo pipefail

# =====================================
# ENVIRONMENT
# =====================================

module load python/3.10 scipy-stack
source /home/vant7ee/rsa/bin/activate

cd /home/vant7ee/scratch/RecognitionData

mkdir -p /home/vant7ee/scratch/logs/pairwise_raw

# =====================================
# CHUNK SETTINGS
# =====================================

CHUNK_SIZE=200

PAIR_START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
PAIR_END=$((PAIR_START + CHUNK_SIZE))

RESULT_DIR=/home/vant7ee/scratch/RecognitionData/results/framework_pairwise_raw_all

mkdir -p ${RESULT_DIR}

DONE_FLAG=${RESULT_DIR}/chunk_${PAIR_START}_${PAIR_END}.done

# =====================================
# SKIP COMPLETED CHUNKS
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
echo "PAIRWISE RAW DECODING"
echo "====================================="
echo "SLURM JOB ID: ${SLURM_JOB_ID}"
echo "ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"

echo "PAIR START: ${PAIR_START}"
echo "PAIR END: ${PAIR_END}"

echo "CHUNK SIZE: ${CHUNK_SIZE}"

echo "TIME LIMIT: 03:00:00"
echo "CPUS: 2"
echo "MEMORY: 16G"
echo "====================================="

# =====================================
# RUN DECODING
# =====================================

python -m svm.scripts.run_pairwise_decoding \
  --manifest /home/vant7ee/scratch/RecognitionData/manifests/raw_manifest.csv \
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