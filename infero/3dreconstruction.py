"""
PLEASE USE THIS SCRIPT AFTER YOU APPRORIATELY INSTALLED https://github.com/radekd91/inferno and making sure you have GPU to run this code. you can use run.sh to setup the processing.**
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import inferno
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse

from inferno_apps.FaceReconstruction.utils.output import (
    save_obj,
    save_images,
    save_codes,
)

from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets


def resolve_flame_decoder(face_rec_model):
    """
    Resolve the actual FLAME decoder inside the outer
    FaceReconstructionBase wrapper.

    Inferno's demo loads FaceReconstructionBase, whose actual shape
    model lives under face_rec_model.shape_model.
    """

    candidates = []

    # Most likely for the current FaceReconstructionBase architecture.
    if hasattr(face_rec_model, "shape_model"):
        shape_model = face_rec_model.shape_model

        if hasattr(shape_model, "flame"):
            candidates.append(
                (
                    "face_rec_model.shape_model.flame",
                    shape_model.flame,
                )
            )

        if hasattr(shape_model, "deca"):
            if hasattr(shape_model.deca, "flame"):
                candidates.append(
                    (
                        "face_rec_model.shape_model.deca.flame",
                        shape_model.deca.flame,
                    )
                )

    # Diagnostic fallback: do not silently guess.
    if len(candidates) == 0:
        shape_model_type = (
            type(face_rec_model.shape_model).__name__
            if hasattr(face_rec_model, "shape_model")
            else "MISSING"
        )

        shape_model_attrs = (
            sorted(
                x
                for x in dir(face_rec_model.shape_model)
                if not x.startswith("_")
            )
            if hasattr(face_rec_model, "shape_model")
            else []
        )

        raise RuntimeError(
            "\nCould not locate FLAME decoder.\n"
            f"face_rec_model type: {type(face_rec_model).__name__}\n"
            f"shape_model type: {shape_model_type}\n"
            f"shape_model public attrs: {shape_model_attrs}\n"
        )

    # If multiple valid paths exist, they should normally point to the
    # same decoder. Use the first and report exactly what was selected.
    path, flame_decoder = candidates[0]

    print("=" * 79)
    print("FLAME DECODER RESOLVED")
    print("=" * 79)
    print("Path:", path)
    print("Class:", type(flame_decoder).__name__)
    print("=" * 79)

    return flame_decoder


def main():
    parser = argparse.ArgumentParser()

    # ============================================================
    # ARGUMENTS
    # ============================================================

    parser.add_argument(
        "--input_folder",
        type=str,
        default=str(
            Path(get_path_to_assets())
            / "data/EMOCA_test_example_data/images/affectnet_test_examples"
        ),
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="image_output",
        help="Output folder to save the results to.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="EMICA-CVT_flame2020_notexture",
        help="Name of the model to use.",
    )

    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="EMICA_flame2020_notexture",
    # )

    parser.add_argument(
        "--path_to_models",
        type=str,
        default=str(
            Path(get_path_to_assets())
            / "FaceReconstruction/models"
        ),
    )

    parser.add_argument(
        "--save_images",
        type=bool,
        default=True,
        help="If true, output images will be saved",
    )

    parser.add_argument(
        "--save_codes",
        type=bool,
        default=False,
        help="If true, output FLAME values for shape, expression, jaw pose will be saved",
    )

    parser.add_argument(
        "--save_mesh",
        type=bool,
        default=False,
        help="If true, output meshes will be saved",
    )

    args = parser.parse_args()


    # ============================================================
    # PATHS / SETTINGS
    # ============================================================

    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name


    # ============================================================
    # 1) LOAD MODEL
    # ============================================================

    print("=" * 79)
    print("LOADING MODEL")
    print("=" * 79)

    face_rec_model, conf = load_model(
        path_to_models,
        model_name,
    )

    face_rec_model.cuda()
    face_rec_model.eval()
    
    # ============================================================
    # RESOLVE ACTUAL FLAME DECODER
    # ============================================================

    flame_decoder = resolve_flame_decoder(
        face_rec_model
    )


    # ============================================================
    # 2) CREATE DATASET
    # ============================================================

    print()
    print("=" * 79)
    print("LOADING DATASET")
    print("=" * 79)

    dataset = TestData(
        input_folder,
        face_detector="fan",
        max_detection=20,
    )

    print(
        f"Total dataset entries: {len(dataset)}"
    )


    # ============================================================
    # 3) RUN MODEL
    # ============================================================

    print()
    print("=" * 79)
    print("RUNNING FACE RECONSTRUCTION")
    print("=" * 79)


    for i in auto.tqdm(
        range(len(dataset))
    ):

        batch = dataset[i]


        # --------------------------------------------------------
        # Standard Inferno / EMICA inference
        # --------------------------------------------------------

        vals = test(
            face_rec_model,
            batch,
        )

        print(
            "VALS KEYS:",
            sorted(vals.keys())
        )

        print(
            "SHAPES:",
            "shapecode =", vals["shapecode"].shape,
            "expcode =", vals["expcode"].shape,
            "globalpose =", vals["globalpose"].shape,
            "jawpose =", vals["jawpose"].shape,
        )


        # ========================================================
        # NEW:
        # CANONICAL IDENTITY-ONLY FLAME SURFACE
        #
        # vals["shapecode"] is MICA's predicted FLAME identity code.
        #
        # Standard mesh_coarse.obj is generated using:
        #
        #   FLAME(
        #       shape_params=shapecode,
        #       expression_params=expcode,
        #       pose_params=posecode
        #   )
        #
        # Here we explicitly set expression and pose to zero:
        #
        #   canonical_identity_vertices =
        #       FLAME(
        #           shape_params=MICA shape,
        #           expression_params=0,
        #           pose_params=0
        #       )
        #
        # This gives identity-only canonical FLAME geometry.
        # ========================================================

        # ========================================================
        # CANONICAL NEUTRAL IDENTITY-ONLY FLAME SURFACE
        # ========================================================

        required_keys = [
            "shapecode",
            "expcode",
            "globalpose",
            "jawpose",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in vals
        ]

        if missing_keys:
            raise KeyError(
                f"Missing required vals keys: {missing_keys}. "
                f"Available keys: {sorted(vals.keys())}"
            )

        print(
            "SHAPES:",
            "shapecode =", vals["shapecode"].shape,
            "expcode =", vals["expcode"].shape,
            "globalpose =", vals["globalpose"].shape,
            "jawpose =", vals["jawpose"].shape,
        )

        if vals["globalpose"].shape[-1] != 3:
            raise RuntimeError(
                "Expected globalpose dimension 3, "
                f"got {vals['globalpose'].shape}"
            )

        if vals["jawpose"].shape[-1] != 3:
            raise RuntimeError(
                "Expected jawpose dimension 3, "
                f"got {vals['jawpose'].shape}"
            )


        with torch.no_grad():

            # ----------------------------------------------------
            # Expression = zero
            # ----------------------------------------------------

            zero_expression = torch.zeros_like(
                vals["expcode"]
            )


            # ----------------------------------------------------
            # Global head pose = zero
            # Jaw pose = zero
            #
            # DecaFLAME expects pose_params with 6 dimensions:
            #
            #     [global XYZ, jaw XYZ]
            #
            # Neck and eye poses are fixed zero parameters inside
            # DecaFLAME, as confirmed from DecaFLAME.py.
            # ----------------------------------------------------

            zero_globalpose = torch.zeros_like(
                vals["globalpose"]
            )

            zero_jawpose = torch.zeros_like(
                vals["jawpose"]
            )

            zero_pose = torch.cat(
                [
                    zero_globalpose,
                    zero_jawpose,
                ],
                dim=-1,
            )

            if zero_pose.shape[-1] != 6:
                raise RuntimeError(
                    "Expected FLAME pose dimension 6, "
                    f"got {zero_pose.shape}"
                )


            # ----------------------------------------------------
            # Decode MICA identity shape into canonical neutral
            # FLAME surface.
            # ----------------------------------------------------

            flame_output = flame_decoder(
                shape_params=vals["shapecode"],
                expression_params=zero_expression,
                pose_params=zero_pose,
            )


            # ----------------------------------------------------
            # Standard FLAME returns:
            #
            #   verts, landmarks2d, landmarks3d
            #
            # FLAME_mediapipe can return:
            #
            #   verts, landmarks2d, landmarks3d,
            #   landmarks2d_mediapipe
            #
            # We only need vertices.
            # ----------------------------------------------------

            if not isinstance(
                flame_output,
                (tuple, list)
            ):
                raise RuntimeError(
                    "Unexpected FLAME output type: "
                    f"{type(flame_output)}"
                )

            if len(flame_output) not in (3, 4):
                raise RuntimeError(
                    "Expected FLAME to return 3 or 4 outputs, "
                    f"got {len(flame_output)}"
                )

            canonical_verts = flame_output[0]


            # ----------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------

            print(
                "zero_pose shape:",
                zero_pose.shape
            )

            print(
                "FLAME output count:",
                len(flame_output)
            )

            print(
                "canonical_verts shape:",
                canonical_verts.shape
            )

            if canonical_verts.ndim != 3:
                raise RuntimeError(
                    "Expected canonical vertices [B, V, 3], "
                    f"got {canonical_verts.shape}"
                )

            if canonical_verts.shape[-1] != 3:
                raise RuntimeError(
                    "Expected final XYZ dimension = 3, "
                    f"got {canonical_verts.shape}"
                )
                
                
            # ----------------------------------------------------
            # Standard FLAME returns:
            #
            #   verts, landmarks2d, landmarks3d
            #
            # FLAME_mediapipe can return:
            #
            #   verts, landmarks2d, landmarks3d,
            #   landmarks2d_mediapipe
            #
            # We only need vertices.
            # ----------------------------------------------------

            if not isinstance(
                flame_output,
                (tuple, list)
            ):
                raise RuntimeError(
                    "Unexpected FLAME output type: "
                    f"{type(flame_output)}"
                )

            if len(flame_output) not in (3, 4):
                raise RuntimeError(
                    "Expected FLAME to return 3 or 4 outputs, "
                    f"got {len(flame_output)}"
                )

            canonical_verts = flame_output[0]


            # ----------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------

            print(
                "zero_pose shape:",
                zero_pose.shape
            )

            print(
                "FLAME output count:",
                len(flame_output)
            )

            print(
                "canonical_verts shape:",
                canonical_verts.shape
            )

            if canonical_verts.ndim != 3:
                raise RuntimeError(
                    "Expected canonical vertices [B, V, 3], "
                    f"got {canonical_verts.shape}"
                )

            if canonical_verts.shape[-1] != 3:
                raise RuntimeError(
                    "Expected final XYZ dimension = 3, "
                    f"got {canonical_verts.shape}"
                )


        # --------------------------------------------------------
        # Standard visualization
        # --------------------------------------------------------

        visdict = face_rec_model.visualize_batch(
            batch,
            i,
            None,
            in_batch_idx=None,
        )


        current_bs = (
            batch["image"].shape[0]
        )


        # ========================================================
        # 4) SAVE EACH DETECTED FACE
        # ========================================================

        for j in range(
            current_bs
        ):

            name = (
                batch["image_name"][j]
            )

            sample_output_folder = (
                Path(output_folder)
                /
                name
            )

            sample_output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )


            # ====================================================
            # NEW:
            # SAVE CANONICAL IDENTITY-ONLY VERTICES
            #
            # Expected:
            #     shape = (5023, 3)
            #
            # These vertices:
            # - contain MICA identity shape
            # - do NOT contain expression
            # - do NOT contain jaw/global pose
            # - do NOT contain camera transform
            # ====================================================

            canonical_np = (
                canonical_verts[j]
                .detach()
                .cpu()
                .numpy()
            )

            if canonical_np.shape != (5023, 3):
                raise RuntimeError(
                    f"{name}: expected canonical surface "
                    f"(5023, 3), got {canonical_np.shape}"
                )

            if not np.isfinite(
                canonical_np
            ).all():

                raise RuntimeError(
                    f"{name}: canonical surface "
                    "contains NaN or Inf."
                )

            canonical_output_path = (
                sample_output_folder
                /
                "canonical_identity_vertices.npy"
            )

            np.save(
                canonical_output_path,
                canonical_np,
            )

            print(
                f"\n{name}: "
                f"canonical_identity_vertices "
                f"{canonical_np.shape} "
                f"saved to {canonical_output_path}"
            )

            # ====================================================
            # ORIGINAL OUTPUTS
            # ====================================================

            if args.save_mesh:

                save_obj(
                    face_rec_model,
                    str(
                        sample_output_folder
                        /
                        "mesh_coarse.obj"
                    ),
                    vals,
                    j,
                )


            if args.save_codes:

                save_codes(
                    Path(
                        output_folder
                    ),
                    name,
                    vals,
                    i=j,
                )


            if args.save_images:

                save_images(
                    output_folder,
                    name,
                    visdict,
                    with_detection=True,
                    i=j,
                )


    # ============================================================
    # FINISH
    # ============================================================

    print()
    print("=" * 79)
    print("DONE")
    print("=" * 79)

    print(
        "Canonical identity-only vertices were saved as:"
    )

    print(
        "<output_folder>/<face>/"
        "canonical_identity_vertices.npy"
    )


if __name__ == "__main__":
    main()
