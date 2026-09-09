set -euo pipefail

# ============================================================
# MODULES + ENV
# ============================================================

module load python/3.10
module load scipy-stack
module load opencv/4.9.0
module load ffmpeg

source inferno_env/bin/activate

export TORCH_HOME=/home/vant7ee/.cache/torch

cd inferno/


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR=/FacerecognitionData/images
INPUT_DIR=/FacerecognitionData/inferno_input
OUTPUT_DIR=/FacerecognitionData/inferno_output

mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"


# ============================================================
# PREPARE 60 IMAGES
# ============================================================

rm -f "$INPUT_DIR"/*

for i in $(seq 1 60); do

    IMAGE="$SOURCE_DIR/$i.png"

    if [[ ! -f "$IMAGE" ]]; then
        echo "ERROR: Missing image: $IMAGE"
        exit 1
    fi

    cp "$IMAGE" "$INPUT_DIR/"

done


echo "============================================================"
echo "INPUT CHECK"
echo "============================================================"

N_IMAGES=$(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.png" | wc -l)

echo "Number of input images: $N_IMAGES"

if [[ "$N_IMAGES" -ne 60 ]]; then
    echo "ERROR: Expected 60 images, found $N_IMAGES"
    exit 1
fi


# ============================================================
# DIAGNOSTICS
# ============================================================

echo
echo "============================================================"
echo "INFERNO 60-FACE GPU RUN"
echo "============================================================"

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "TORCH_HOME: $TORCH_HOME"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo

echo "Cached Torch weights:"
ls -lh "$TORCH_HOME/hub/checkpoints/" || true
echo

nvidia-smi


python - <<'PY'

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

print(
    "cuda device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "NONE"
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

PY


# ============================================================
# RUN ALL 60 FACES
# ============================================================

echo
echo "============================================================"
echo "STARTING INFERNO"
echo "============================================================"

python inferno_apps/FaceReconstruction/demo/demo_face_rec_on_images.py \
  --input_folder "$INPUT_DIR" \
  --output_folder "$OUTPUT_DIR" \
  --model_name EMICA-CVT_flame2020_notexture \
  --save_images True \
  --save_codes True \
  --save_mesh True


# ============================================================
# OUTPUT CHECK
# ============================================================

echo
echo "============================================================"
echo "INFERNO FINISHED"
echo "============================================================"

echo
echo "Total number of output files:"
find "$OUTPUT_DIR" -type f | wc -l


echo
echo "Canonical identity vertex files:"

N_CANONICAL=$(
    find "$OUTPUT_DIR" \
    -type f \
    -name "canonical_identity_vertices.npy" \
    | wc -l
)

echo "$N_CANONICAL"


if [[ "$N_CANONICAL" -ne 60 ]]; then

    echo "ERROR: Expected 60 canonical vertex files, found $N_CANONICAL"

    find "$OUTPUT_DIR" \
        -type f \
        -name "canonical_identity_vertices.npy" \
        | sort

    exit 1

fi


# ============================================================
# VERIFY SHAPES
# ============================================================

python - <<PY

from pathlib import Path
import numpy as np

root = Path("$OUTPUT_DIR")

files = sorted(
    root.rglob("canonical_identity_vertices.npy")
)

print()
print("============================================================")
print("CANONICAL VERTEX VALIDATION")
print("============================================================")

good = 0

for p in files:

    x = np.load(p)

    finite = np.isfinite(x).all()

    print(
        p.parent.name,
        "shape=",
        x.shape,
        "finite=",
        finite
    )

    if x.shape == (5023, 3) and finite:
        good += 1

print()
print("VALID:", good, "/ 60")

if good != 60:
    raise RuntimeError(
        f"Only {good}/60 canonical vertex files are valid."
    )

PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"
