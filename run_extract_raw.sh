#!/bin/bash
#SBATCH --job-name=FEATURE_EXTRACT
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=logs/feature_extract/%x_%j.out
#SBATCH --error=logs/feature_extract/%x_%j.err

set -euo pipefail

module load python/3.10 scipy-stack

# Activate environment
source /path/to/your/env/bin/activate

# Project root
cd /path/to/project

# -----------------------------------------
# CONFIG
# -----------------------------------------

FEATURE_SPACE="raw"
OUT_DIR="features/${FEATURE_SPACE}"

SUBJECTS=(
    10665
)

# -----------------------------------------
# RUN
# -----------------------------------------

for SUB in "${SUBJECTS[@]}"
do

    echo "====================================="
    echo "Subject: ${SUB}"
    echo "Feature Space: ${FEATURE_SPACE}"
    echo "====================================="

    python -m svm.scripts.extract_features \
        --epoch_path epochs/${SUB}-epo.fif \
        --metadata_path epochs/${SUB}-metadata.csv \
        --out_dir ${OUT_DIR} \
        --subject ${SUB} \
        --task recognition \
        --feature_space ${FEATURE_SPACE} \
        --tmin 0.0 \
        --tmax 0.6 \
        --overwrite

done
