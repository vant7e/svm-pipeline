#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build stimulus-visible canonical 3D facial identity RDMs.

Scientific design
-----------------
Visibility and identity geometry are deliberately separated.

VISIBILITY:
    Derived previously from the reconstruction run:

        inferno_output_60

    using:
        - original experimental stimulus aperture
        - z-buffer visibility
        - Inferno front-facing criterion
        - common visibility across faces

    Frozen masks:
        common100 = PRIMARY
        common95  = robustness
        common90  = robustness


IDENTITY GEOMETRY:
    Derived here from the OLD exact MICA shape codes used in the
    previous latent-space MICA analyses:

        inferno_output_60/<face>/shape.npy

    Each old 300-D MICA identity code is decoded through the same
    FLAME_mediapipe decoder with:

        expression = 0
        global pose = 0
        jaw pose = 0

    producing canonical neutral identity-only vertices:

        (60, 5023, 3)


FINAL SURFACE RDM:
    Canonical identity vertices are restricted to the frozen
    stimulus-visible common masks.

    For faces i and j:

        D_ij =
            sqrt(
                sum_v || X_i(v) - X_j(v) ||^2
            )

    This is equivalent to Euclidean distance after flattening the
    corresponding XYZ coordinates of the selected vertices.


PRIMARY:
    common100

ROBUSTNESS:
    common95
    common90


IMPORTANT INTERPRETATION
------------------------
This is:

    MICA-estimated canonical 3D facial identity geometry
    restricted to the stimulus-visible facial surface.

It is NOT:
    - ground-truth 3D depth
    - whole-head geometry
    - posed geometry
    - expression geometry
    - camera geometry


Outputs
-------
canonical_vertices_old_codes.npy
    shape (60, 5023, 3)

pair_table.csv
    canonical scipy.pdist pair order, 1770 rows

mica_latent_shape_code_rdm.npy
    old 300-D MICA coefficient-space Euclidean RDM

mica_whole_canonical_surface_rdm.npy
    whole 5023-vertex canonical surface diagnostic RDM

mica_surface_common100_matrix.npy
mica_surface_common100_rdm.npy

mica_surface_common95_matrix.npy
mica_surface_common95_rdm.npy

mica_surface_common90_matrix.npy
mica_surface_common90_rdm.npy

surface_rdm_correlations.csv
latent_surface_correlations.csv
whole_surface_correlations.csv
surface_rdm_summary.csv

No neural data are used anywhere in this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import (
    pdist,
    squareform,
)

from scipy.stats import (
    spearmanr,
)

from inferno_apps.FaceReconstruction.utils.load import (
    load_model,
)

from inferno.utils.other import (
    get_path_to_assets,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/home/vant7ee/scratch/FacerecognitionData"
)


# OLD exact MICA run used in earlier latent-space analyses.
OLD_MICA_DIR = (
    ROOT
    /
    "inferno_output_60"
)


# Frozen visibility masks generated from the NEW _V run.
VISIBILITY_DIR = (
    ROOT
    /
    "analysis"
    /
    "mica_3d_shape"
    /
    "stimulus_visible_surface"
)


OUT_DIR = (
    VISIBILITY_DIR
    /
    "surface_rdms"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INFERNO MODEL
# =============================================================================

MODEL_NAME = (
    "EMICA-CVT_flame2020_notexture"
)


PATH_TO_MODELS = (
    Path(
        get_path_to_assets()
    )
    /
    "FaceReconstruction"
    /
    "models"
)


# =============================================================================
# CONSTANTS
# =============================================================================

FACE_IDS = list(
    range(
        1,
        61,
    )
)

N_FACES = 60

N_VERTICES = 5023

N_SHAPE = 300

N_EXPRESSION = 100

N_POSE = 6

N_PAIRS = (
    N_FACES
    *
    (
        N_FACES
        -
        1
    )
    //
    2
)

assert N_PAIRS == 1770


MASK_LABELS = [
    "100",
    "95",
    "90",
]


# =============================================================================
# FLAME DECODER RESOLUTION
# =============================================================================

def resolve_flame_decoder(
    face_rec_model,
):
    """
    Resolve the actual FLAME decoder inside Inferno's outer
    FaceReconstructionBase wrapper.

    This is the same logic that was already validated in the
    successful canonical-vertex reconstruction run.
    """

    candidates = []


    if hasattr(
        face_rec_model,
        "shape_model",
    ):

        shape_model = (
            face_rec_model
            .shape_model
        )


        if hasattr(
            shape_model,
            "flame",
        ):

            candidates.append(
                (
                    "face_rec_model.shape_model.flame",
                    shape_model.flame,
                )
            )


        if hasattr(
            shape_model,
            "deca",
        ):

            if hasattr(
                shape_model.deca,
                "flame",
            ):

                candidates.append(
                    (
                        "face_rec_model.shape_model.deca.flame",
                        shape_model.deca.flame,
                    )
                )


    if len(
        candidates
    ) == 0:

        shape_model_type = (
            type(
                face_rec_model.shape_model
            ).__name__
            if hasattr(
                face_rec_model,
                "shape_model",
            )
            else
            "MISSING"
        )


        shape_model_attrs = (
            sorted(
                x
                for x
                in dir(
                    face_rec_model.shape_model
                )
                if not x.startswith(
                    "_"
                )
            )
            if hasattr(
                face_rec_model,
                "shape_model",
            )
            else
            []
        )


        raise RuntimeError(
            "\nCould not locate FLAME decoder.\n"
            f"face_rec_model type: "
            f"{type(face_rec_model).__name__}\n"
            f"shape_model type: "
            f"{shape_model_type}\n"
            f"shape_model public attrs: "
            f"{shape_model_attrs}\n"
        )


    path, flame_decoder = (
        candidates[
            0
        ]
    )


    print(
        "=" * 79
    )

    print(
        "FLAME DECODER RESOLVED"
    )

    print(
        "=" * 79
    )

    print(
        "Path:",
        path,
    )

    print(
        "Class:",
        type(
            flame_decoder
        ).__name__,
    )

    print(
        "=" * 79
    )


    return flame_decoder


# =============================================================================
# LOAD INFERNO MODEL — CPU ONLY
# =============================================================================

def load_inferno_model_cpu():
    """
    Load the same pretrained EMICA model as the successful inference
    script, but keep the complete model on CPU.

    No images are passed through the MICA encoder.
    No CUDA device is required.
    """

    print()

    print(
        "=" * 79
    )

    print(
        "LOAD INFERNO MODEL — CPU ONLY"
    )

    print(
        "=" * 79
    )


    print(
        "Model directory:",
        PATH_TO_MODELS,
    )

    print(
        "Model name:",
        MODEL_NAME,
    )


    # ============================================================
    # FORCE ALL TORCH CHECKPOINTS TO LOAD ON CPU
    #
    # Some Inferno / MICA checkpoints were originally serialized
    # from CUDA. Internally, MICA calls:
    #
    #     torch.load(model_path)
    #
    # without map_location. On a CPU-only compute node this raises:
    #
    #     Attempting to deserialize object on a CUDA device...
    #
    # Temporarily wrap torch.load so that any load call that does
    # NOT already specify map_location is mapped to CPU.
    #
    # Existing explicit map_location arguments are preserved.
    # ============================================================

    original_torch_load = torch.load


    def torch_load_cpu(
        *args,
        **kwargs,
    ):

        if (
            "map_location"
            not in kwargs
            or
            kwargs[
                "map_location"
            ]
            is None
        ):

            kwargs[
                "map_location"
            ] = torch.device(
                "cpu"
            )

        return original_torch_load(
            *args,
            **kwargs,
        )


    torch.load = torch_load_cpu


    try:

        face_rec_model, conf = (
            load_model(
                str(
                    PATH_TO_MODELS
                ),
                MODEL_NAME,
            )
        )

    finally:

        # Restore torch.load immediately after Inferno has finished
        # constructing/loading the model.
        torch.load = (
            original_torch_load
        )


    # Keep the entire loaded model explicitly on CPU.

    face_rec_model = (
        face_rec_model.cpu()
    )

    face_rec_model.eval()


    # IMPORTANT:
    #
    # Do NOT call .cuda().
    #
    # The model is only being used as a container for the FLAME
    # decoder. Move everything explicitly to CPU.

    face_rec_model = (
        face_rec_model.cpu()
    )

    face_rec_model.eval()


    flame_decoder = (
        resolve_flame_decoder(
            face_rec_model
        )
    )


    flame_decoder = (
        flame_decoder.cpu()
    )

    flame_decoder.eval()


    print()

    print(
        "Model device check:"
    )


    parameter = next(
        flame_decoder.parameters(),
        None,
    )


    if parameter is not None:

        print(
            "FLAME parameter device:",
            parameter.device,
        )

    else:

        buffer = next(
            flame_decoder.buffers(),
            None,
        )

        if buffer is not None:

            print(
                "FLAME buffer device:",
                buffer.device,
            )

        else:

            print(
                "FLAME has no parameters/buffers "
                "available for device diagnostic."
            )


    return (
        face_rec_model,
        flame_decoder,
        conf,
    )


# =============================================================================
# LOAD FROZEN VISIBILITY MASKS
# =============================================================================

def load_visibility_masks():

    print()

    print(
        "=" * 79
    )

    print(
        "LOAD FROZEN STIMULUS-VISIBILITY MASKS"
    )

    print(
        "=" * 79
    )


    masks = {}


    expected_counts = {
        "100": 1565,
        "95": 1694,
        "90": 1749,
    }


    for label in MASK_LABELS:

        path = (
            VISIBILITY_DIR
            /
            f"common_visible_mask_{label}.npy"
        )


        if not path.exists():

            raise FileNotFoundError(
                path
            )


        mask = np.load(
            path
        )


        mask = np.asarray(
            mask
        ).astype(
            bool
        )


        if mask.shape != (
            N_VERTICES,
        ):

            raise RuntimeError(
                f"common{label}: "
                f"expected mask shape "
                f"({N_VERTICES},), "
                f"got {mask.shape}"
            )


        n_selected = int(
            mask.sum()
        )


        print(
            f"common{label}: "
            f"{n_selected} / "
            f"{N_VERTICES} vertices"
        )


        # These counts are used as an additional safeguard so we
        # cannot silently analyze a subsequently modified mask.

        expected_n = (
            expected_counts[
                label
            ]
        )


        if n_selected != expected_n:

            raise RuntimeError(
                f"\nFrozen mask mismatch for "
                f"common{label}.\n"
                f"Expected {expected_n} vertices "
                f"from the validated visibility run, "
                f"but found {n_selected}.\n"
                "Do not continue until the mask version "
                "is verified.\n"
            )


        masks[
            label
        ] = mask


    # Nested-mask sanity checks.

    if not np.all(
        masks[
            "100"
        ]
        <=
        masks[
            "95"
        ]
    ):

        raise RuntimeError(
            "common100 is not a subset of common95."
        )


    if not np.all(
        masks[
            "95"
        ]
        <=
        masks[
            "90"
        ]
    ):

        raise RuntimeError(
            "common95 is not a subset of common90."
        )


    print(
        "Mask nesting: PASS"
    )


    return masks


# =============================================================================
# LOAD OLD EXACT MICA SHAPE CODES
# =============================================================================

def load_old_shape_codes():

    print()

    print(
        "=" * 79
    )

    print(
        "LOAD OLD EXACT MICA SHAPE CODES"
    )

    print(
        "=" * 79
    )


    rows = []


    for face_id in FACE_IDS:

        folder = (
            OLD_MICA_DIR
            /
            str(
                face_id
                *
                100
            )
        )


        path = (
            folder
            /
            "shape.npy"
        )


        if not path.exists():

            raise FileNotFoundError(
                path
            )


        shape = np.load(
            path
        )


        shape = np.asarray(
            shape,
            dtype=np.float32,
        ).reshape(
            -1
        )


        if shape.shape != (
            N_SHAPE,
        ):

            raise RuntimeError(
                f"Face {face_id:02d}: "
                f"expected 300-D shape code, "
                f"got {shape.shape}"
            )


        if not np.isfinite(
            shape
        ).all():

            raise RuntimeError(
                f"Face {face_id:02d}: "
                "shape code contains NaN/Inf."
            )


        rows.append(
            shape
        )


    shape_codes = np.stack(
        rows,
        axis=0,
    )


    if shape_codes.shape != (
        N_FACES,
        N_SHAPE,
    ):

        raise RuntimeError(
            f"Unexpected stacked shape-code "
            f"matrix: {shape_codes.shape}"
        )


    print(
        "Shape-code matrix:",
        shape_codes.shape,
    )


    print(
        "Finite:",
        bool(
            np.isfinite(
                shape_codes
            ).all()
        ),
    )


    return shape_codes


# =============================================================================
# DECODE OLD CODES TO CANONICAL NEUTRAL VERTICES
# =============================================================================

def decode_canonical_vertices(
    flame_decoder,
    shape_codes,
):
    """
    Decode the OLD exact MICA shape codes through the same
    FLAME_mediapipe decoder.

    Explicitly:
        expression = 0
        global pose = 0
        jaw pose = 0

    Neck and eye poses use FLAME's internal neutral/default values,
    which were previously confirmed to be zero in this configuration.
    """

    print()

    print(
        "=" * 79
    )

    print(
        "DECODE OLD CODES → CANONICAL NEUTRAL FLAME SURFACE"
    )

    print(
        "=" * 79
    )


    device = torch.device(
        "cpu"
    )


    shape_tensor = torch.tensor(
        shape_codes,
        dtype=torch.float32,
        device=device,
    )


    zero_expression = torch.zeros(
        (
            N_FACES,
            N_EXPRESSION,
        ),
        dtype=torch.float32,
        device=device,
    )


    zero_pose = torch.zeros(
        (
            N_FACES,
            N_POSE,
        ),
        dtype=torch.float32,
        device=device,
    )


    print(
        "shape_tensor:",
        shape_tensor.shape,
    )

    print(
        "zero_expression:",
        zero_expression.shape,
    )

    print(
        "zero_pose:",
        zero_pose.shape,
    )


    with torch.no_grad():

        flame_output = (
            flame_decoder(
                shape_params=shape_tensor,
                expression_params=zero_expression,
                pose_params=zero_pose,
            )
        )


    if not isinstance(
        flame_output,
        (
            tuple,
            list,
        ),
    ):

        raise RuntimeError(
            "Unexpected FLAME output type: "
            f"{type(flame_output)}"
        )


    if len(
        flame_output
    ) not in (
        3,
        4,
    ):

        raise RuntimeError(
            "Expected FLAME to return 3 or 4 "
            f"objects, got {len(flame_output)}"
        )


    canonical_verts = (
        flame_output[
            0
        ]
    )


    print(
        "FLAME output count:",
        len(
            flame_output
        ),
    )


    print(
        "canonical_verts tensor:",
        canonical_verts.shape,
    )


    if canonical_verts.shape != (
        N_FACES,
        N_VERTICES,
        3,
    ):

        raise RuntimeError(
            "Expected canonical vertices "
            f"(60, 5023, 3), "
            f"got {canonical_verts.shape}"
        )


    canonical_np = (
        canonical_verts
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


    if not np.isfinite(
        canonical_np
    ).all():

        raise RuntimeError(
            "Canonical vertex array "
            "contains NaN or Inf."
        )


    print(
        "Canonical array:",
        canonical_np.shape,
    )


    print(
        "Finite:",
        bool(
            np.isfinite(
                canonical_np
            ).all()
        ),
    )


    return canonical_np


# =============================================================================
# OPTIONAL DIAGNOSTIC:
# OLD-CODE CANONICAL vs NEW _V CANONICAL
# =============================================================================

def compare_old_decoded_vs_new_vertices(
    canonical_old,
):

    new_dir = (
        ROOT
        /
        "inferno_output_60_V"
    )


    rows = []


    for idx, face_id in enumerate(
        FACE_IDS
    ):

        path = (
            new_dir
            /
            str(
                face_id
                *
                100
            )
            /
            "canonical_identity_vertices.npy"
        )


        if not path.exists():

            continue


        new_vertices = np.load(
            path
        )


        if new_vertices.shape != (
            N_VERTICES,
            3,
        ):

            continue


        diff = (
            canonical_old[
                idx
            ]
            -
            new_vertices
        )


        per_vertex_distance = np.linalg.norm(
            diff,
            axis=1,
        )


        rows.append(
            {
                "face_id":
                    face_id,

                "mean_abs_coordinate_difference":
                    float(
                        np.mean(
                            np.abs(
                                diff
                            )
                        )
                    ),

                "max_abs_coordinate_difference":
                    float(
                        np.max(
                            np.abs(
                                diff
                            )
                        )
                    ),

                "mean_vertex_euclidean_difference":
                    float(
                        np.mean(
                            per_vertex_distance
                        )
                    ),

                "max_vertex_euclidean_difference":
                    float(
                        np.max(
                            per_vertex_distance
                        )
                    ),
            }
        )


    df = pd.DataFrame(
        rows
    )


    if len(
        df
    ) > 0:

        df.to_csv(
            OUT_DIR
            /
            "old_decoded_vs_new_canonical_vertices.csv",
            index=False,
        )


        print()

        print(
            "=" * 79
        )

        print(
            "OLD-CODE vs NEW _V CANONICAL DIAGNOSTIC"
        )

        print(
            "=" * 79
        )


        print(
            "Mean coordinate difference:",
            df[
                "mean_abs_coordinate_difference"
            ].mean(),
        )


        print(
            "Maximum coordinate difference:",
            df[
                "max_abs_coordinate_difference"
            ].max(),
        )


    return df


# =============================================================================
# PAIR TABLE
# =============================================================================

def build_pair_table():

    rows = []

    pair_index = 0


    # scipy.pdist order:
    #
    # (0,1), (0,2), ...,
    # (1,2), (1,3), ...
    #
    # For FACE_IDS 1..60 this is equivalent to canonical upper
    # triangle face-ID order.

    for i in range(
        N_FACES
    ):

        for j in range(
            i + 1,
            N_FACES,
        ):

            face_a = (
                FACE_IDS[
                    i
                ]
            )

            face_b = (
                FACE_IDS[
                    j
                ]
            )


            rows.append(
                {
                    "canonical_pair_index":
                        pair_index,

                    "face_a":
                        face_a,

                    "face_b":
                        face_b,

                    "face_set_pair":
                        (
                            "Asian-Asian"
                            if (
                                face_a <= 30
                                and
                                face_b <= 30
                            )
                            else
                            "White-White"
                            if (
                                face_a >= 31
                                and
                                face_b >= 31
                            )
                            else
                            "Cross-race"
                        ),
                }
            )


            pair_index += 1


    df = pd.DataFrame(
        rows
    )


    if len(
        df
    ) != N_PAIRS:

        raise RuntimeError(
            f"Pair table has "
            f"{len(df)} rows, "
            f"expected {N_PAIRS}."
        )


    return df


# =============================================================================
# RDM HELPERS
# =============================================================================

def build_flattened_surface_rdm(
    vertices,
    mask=None,
):
    """
    Build Euclidean RDM from corresponding XYZ surface coordinates.

    If mask is None:
        use all 5023 vertices.

    Otherwise:
        use only the selected common stimulus-visible vertices.
    """

    if mask is None:

        selected = (
            vertices
        )

    else:

        selected = (
            vertices[
                :,
                mask,
                :
            ]
        )


    if selected.shape[
        0
    ] != N_FACES:

        raise RuntimeError(
            "Unexpected face dimension."
        )


    X = selected.reshape(
        N_FACES,
        -1,
    )


    rdm = pdist(
        X,
        metric="euclidean",
    )


    matrix = squareform(
        rdm
    )


    if rdm.shape != (
        N_PAIRS,
    ):

        raise RuntimeError(
            f"Unexpected RDM shape: "
            f"{rdm.shape}"
        )


    if matrix.shape != (
        N_FACES,
        N_FACES,
    ):

        raise RuntimeError(
            f"Unexpected matrix shape: "
            f"{matrix.shape}"
        )


    if not np.isfinite(
        rdm
    ).all():

        raise RuntimeError(
            "RDM contains NaN/Inf."
        )


    return (
        matrix,
        rdm,
        selected.shape[
            1
        ],
    )


def build_latent_rdm(
    shape_codes,
):

    rdm = pdist(
        shape_codes,
        metric="euclidean",
    )


    if rdm.shape != (
        N_PAIRS,
    ):

        raise RuntimeError(
            f"Unexpected latent RDM "
            f"shape: {rdm.shape}"
        )


    return rdm


# =============================================================================
# RANK-Z — OPTIONAL SAVED REPRESENTATION
# =============================================================================

def rank_z(
    x,
):
    """
    Rank-transform and z-score an RDM.

    Saved because downstream RSA commonly operates on rank-z stimulus
    RDMs, while raw distances are also retained.
    """

    x = np.asarray(
        x,
        dtype=float,
    )


    ranks = pd.Series(
        x
    ).rank(
        method="average"
    ).to_numpy(
        dtype=float
    )


    sd = ranks.std(
        ddof=1
    )


    if sd <= 0:

        raise RuntimeError(
            "Cannot rank-z a constant RDM."
        )


    return (
        ranks
        -
        ranks.mean()
    ) / sd


# =============================================================================
# CORRELATION
# =============================================================================

def spearman(
    a,
    b,
):

    result = spearmanr(
        a,
        b,
    )


    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 79
    )

    print(
        "STIMULUS-VISIBLE CANONICAL 3D FACIAL IDENTITY RDM"
    )

    print(
        "=" * 79
    )


    # ============================================================
    # 1. FROZEN VISIBILITY MASKS
    # ============================================================

    masks = (
        load_visibility_masks()
    )


    # ============================================================
    # 2. OLD EXACT SHAPE CODES
    # ============================================================

    shape_codes = (
        load_old_shape_codes()
    )


    np.save(
        OUT_DIR
        /
        "old_exact_mica_shape_codes.npy",
        shape_codes,
    )


    # ============================================================
    # 3. LOAD FLAME DECODER ON CPU
    # ============================================================

    (
        face_rec_model,
        flame_decoder,
        conf,
    ) = load_inferno_model_cpu()


    # ============================================================
    # 4. DECODE OLD CODES TO CANONICAL NEUTRAL SURFACE
    # ============================================================

    canonical_vertices = (
        decode_canonical_vertices(
            flame_decoder,
            shape_codes,
        )
    )


    np.save(
        OUT_DIR
        /
        "canonical_vertices_old_codes.npy",
        canonical_vertices,
    )


    # Comparison with the new _V rerun is diagnostic only.

    compare_old_decoded_vs_new_vertices(
        canonical_vertices
    )


    # ============================================================
    # 5. PAIR TABLE
    # ============================================================

    pair_table = (
        build_pair_table()
    )


    pair_table.to_csv(
        OUT_DIR
        /
        "pair_table.csv",
        index=False,
    )


    # ============================================================
    # 6. OLD LATENT MICA RDM
    # ============================================================

    latent_rdm = (
        build_latent_rdm(
            shape_codes
        )
    )


    np.save(
        OUT_DIR
        /
        "mica_latent_shape_code_rdm.npy",
        latent_rdm,
    )


    np.save(
        OUT_DIR
        /
        "mica_latent_shape_code_rdm_rankz.npy",
        rank_z(
            latent_rdm
        ),
    )


    # ============================================================
    # 7. WHOLE CANONICAL SURFACE DIAGNOSTIC
    # ============================================================

    (
        whole_matrix,
        whole_rdm,
        n_whole,
    ) = build_flattened_surface_rdm(
        canonical_vertices,
        mask=None,
    )


    np.save(
        OUT_DIR
        /
        "mica_whole_canonical_surface_matrix.npy",
        whole_matrix,
    )


    np.save(
        OUT_DIR
        /
        "mica_whole_canonical_surface_rdm.npy",
        whole_rdm,
    )


    np.save(
        OUT_DIR
        /
        "mica_whole_canonical_surface_rdm_rankz.npy",
        rank_z(
            whole_rdm
        ),
    )


    # ============================================================
    # 8. STIMULUS-VISIBLE SURFACE RDMs
    # ============================================================

    surface_rdms = {}

    summary_rows = []


    for label in MASK_LABELS:

        print()

        print(
            "=" * 79
        )

        print(
            f"BUILD COMMON {label}% "
            "STIMULUS-VISIBLE SURFACE RDM"
        )

        print(
            "=" * 79
        )


        (
            matrix,
            rdm,
            n_selected,
        ) = build_flattened_surface_rdm(
            canonical_vertices,
            mask=masks[
                label
            ],
        )


        surface_rdms[
            label
        ] = rdm


        np.save(
            OUT_DIR
            /
            f"mica_surface_common{label}_matrix.npy",
            matrix,
        )


        np.save(
            OUT_DIR
            /
            f"mica_surface_common{label}_rdm.npy",
            rdm,
        )


        np.save(
            OUT_DIR
            /
            f"mica_surface_common{label}_rdm_rankz.npy",
            rank_z(
                rdm
            ),
        )


        print(
            "Selected vertices:",
            n_selected,
        )


        print(
            "RDM shape:",
            rdm.shape,
        )


        print(
            "RDM min:",
            float(
                rdm.min()
            ),
        )


        print(
            "RDM mean:",
            float(
                rdm.mean()
            ),
        )


        print(
            "RDM max:",
            float(
                rdm.max()
            ),
        )


        summary_rows.append(
            {
                "surface":
                    f"common{label}",

                "n_vertices":
                    n_selected,

                "n_pairs":
                    len(
                        rdm
                    ),

                "rdm_min":
                    float(
                        rdm.min()
                    ),

                "rdm_mean":
                    float(
                        rdm.mean()
                    ),

                "rdm_sd":
                    float(
                        rdm.std(
                            ddof=1
                        )
                    ),

                "rdm_max":
                    float(
                        rdm.max()
                    ),
            }
        )


    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUT_DIR
        /
        "surface_rdm_summary.csv",
        index=False,
    )


    # ============================================================
    # 9. THRESHOLD ROBUSTNESS
    # ============================================================

    threshold_rows = []


    threshold_pairs = [
        (
            "100",
            "95",
        ),
        (
            "100",
            "90",
        ),
        (
            "95",
            "90",
        ),
    ]


    print()

    print(
        "=" * 79
    )

    print(
        "SURFACE MASK THRESHOLD ROBUSTNESS"
    )

    print(
        "=" * 79
    )


    for a, b in threshold_pairs:

        rho, p = spearman(
            surface_rdms[
                a
            ],
            surface_rdms[
                b
            ],
        )


        print(
            f"common{a} vs common{b}: "
            f"rho={rho:.6f}, "
            f"p={p:.3e}"
        )


        threshold_rows.append(
            {
                "rdm_a":
                    f"common{a}",

                "rdm_b":
                    f"common{b}",

                "spearman_rho":
                    rho,

                "p_value":
                    p,
            }
        )


    pd.DataFrame(
        threshold_rows
    ).to_csv(
        OUT_DIR
        /
        "surface_rdm_correlations.csv",
        index=False,
    )


    # ============================================================
    # 10. OLD LATENT vs VISIBLE SURFACE
    # ============================================================

    latent_rows = []


    print()

    print(
        "=" * 79
    )

    print(
        "OLD LATENT MICA vs STIMULUS-VISIBLE SURFACE"
    )

    print(
        "=" * 79
    )


    for label in MASK_LABELS:

        rho, p = spearman(
            latent_rdm,
            surface_rdms[
                label
            ],
        )


        print(
            f"latent vs common{label}: "
            f"rho={rho:.6f}, "
            f"p={p:.3e}"
        )


        latent_rows.append(
            {
                "surface_mask":
                    f"common{label}",

                "spearman_rho_with_latent":
                    rho,

                "p_value":
                    p,
            }
        )


    pd.DataFrame(
        latent_rows
    ).to_csv(
        OUT_DIR
        /
        "latent_surface_correlations.csv",
        index=False,
    )


    # ============================================================
    # 11. WHOLE CANONICAL vs VISIBLE SURFACE
    # ============================================================

    whole_rows = []


    print()

    print(
        "=" * 79
    )

    print(
        "WHOLE CANONICAL vs STIMULUS-VISIBLE SURFACE"
    )

    print(
        "=" * 79
    )


    for label in MASK_LABELS:

        rho, p = spearman(
            whole_rdm,
            surface_rdms[
                label
            ],
        )


        print(
            f"whole vs common{label}: "
            f"rho={rho:.6f}, "
            f"p={p:.3e}"
        )


        whole_rows.append(
            {
                "surface_mask":
                    f"common{label}",

                "spearman_rho_with_whole_surface":
                    rho,

                "p_value":
                    p,
            }
        )


    pd.DataFrame(
        whole_rows
    ).to_csv(
        OUT_DIR
        /
        "whole_surface_correlations.csv",
        index=False,
    )


    # ============================================================
    # 12. PRIMARY RDM COPY / EXPLICIT LABEL
    # ============================================================

    primary_rdm = (
        surface_rdms[
            "100"
        ]
    )


    np.save(
        OUT_DIR
        /
        "PRIMARY_mica_stimulus_visible_3d_rdm.npy",
        primary_rdm,
    )


    np.save(
        OUT_DIR
        /
        "PRIMARY_mica_stimulus_visible_3d_rdm_rankz.npy",
        rank_z(
            primary_rdm
        ),
    )


    # ============================================================
    # FINISH
    # ============================================================

    print()

    print(
        "=" * 79
    )

    print(
        "DONE"
    )

    print(
        "=" * 79
    )


    print(
        "PRIMARY MODEL:"
    )

    print(
        "  common100 stimulus-visible "
        "canonical 3D facial identity surface"
    )


    print()

    print(
        "ROBUSTNESS MODELS:"
    )

    print(
        "  common95"
    )

    print(
        "  common90"
    )


    print()

    print(
        "Primary RDM:"
    )

    print(
        OUT_DIR
        /
        "PRIMARY_mica_stimulus_visible_3d_rdm.npy"
    )


    print()

    print(
        "Output directory:"
    )

    print(
        OUT_DIR
    )

    print(
        "=" * 79
    )


if __name__ == "__main__":

    main()
