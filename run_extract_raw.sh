#!/bin/bash
#SBATCH --job-name=RAW_EXTRACT
#SBATCH --account=def-jmeltzer

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=/home/vant7ee/scratch/logs/raw_extract/raw_%j.out
#SBATCH --error=/home/vant7ee/scratch/logs/raw_extract/raw_%j.err

set -euo pipefail

module load python/3.10 scipy-stack
source /home/vant7ee/rsa/bin/activate

cd /home/vant7ee/scratch/RecognitionData

SUBJECTS=(
    10665
    20632
    20775
    20845
    20902
    21019
    21034
    21047
    21104
    21105
    21124
    21143
    21144
    21165
    21194
    21212
    21214
    21224
)

for SUB in "${SUBJECTS[@]}"
do

    echo "====================================="
    echo "Subject: ${SUB}"
    echo "====================================="

    python -m svm.scripts.extract_features \
        --epoch_path /home/vant7ee/scratch/RecognitionData/epochs/concat_common325/${SUB}-epo.fif \
        --metadata_path /home/vant7ee/scratch/RecognitionData/epochs/concat_common325/${SUB}-metadata.csv \
        --out_dir /home/vant7ee/scratch/RecognitionData/features/raw \
        --subject ${SUB} \
        --task recognition \
        --feature_space raw \
        --tmin 0.0 \
        --tmax 0.6 \
        --overwrite

done