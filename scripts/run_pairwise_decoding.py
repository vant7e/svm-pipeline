# svm/scripts/run_pairwise_decoding.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from svm.core.io import load_feature_package

from svm.core.manifest import (
    load_manifest,
    build_manifest_from_feature_dir,
    validate_manifest_feature_consistency,
)

from svm.core.labels import (
    build_pairs_from_column,
    sample_pairs,
    chunk_pairs,
    validate_trial_alignment,
)

from svm.core.spatial import build_ctf_roi_map

from svm.decode.projection import (
    project_multiple_subjects_to_X,
)

from svm.decode.pairwise import (
    run_pairwise_loso_decoding,
    run_pairwise_single_subject_cv_decoding,
)

from svm.decode.splits import (
    make_leave_one_group_out_splits,
    make_leave_one_run_out_splits,
    make_leave_one_block_out_splits,
    make_stratified_kfold_splits,
)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_spatial_values(values):

    if values is None or values.strip() == "":
        return None

    return [
        v.strip()
        for v in values.split(",")
        if v.strip()
    ]


def load_features_from_manifest(manifest, mmap=True):

    feature_list = []
    meta_list = []

    for _, row in manifest.iterrows():

        feature, meta = load_feature_package(
            feature_path=row["feature_path"],
            meta_path=row["meta_path"],
            mmap=mmap,
        )

        feature_list.append(feature)
        meta_list.append(meta)

    return feature_list, meta_list


def metadata_from_meta(meta):

    if "labels" not in meta or meta["labels"] is None:
        raise KeyError(
            "Feature metadata does not contain labels."
        )

    return pd.DataFrame(meta["labels"])


def get_values_list(meta_list, pair_column):

    values_list = []

    for i, meta in enumerate(meta_list):

        if "labels" not in meta or meta["labels"] is None:
            raise KeyError(f"Metadata index {i} has no labels.")

        if pair_column not in meta["labels"]:
            raise KeyError(
                f"pair_column '{pair_column}' not found "
                f"in metadata labels. "
                f"Available columns: "
                f"{list(meta['labels'].keys())}"
            )

        values = np.asarray(
            meta["labels"][pair_column]
        ).astype(str)

        values_list.append(values)

    return values_list


def main():

    parser = argparse.ArgumentParser(
        description=(
            "General pairwise decoding framework "
            "for neural feature packages."
        )
    )

    # -------------------------------------------------
    # Dataset / manifest
    # -------------------------------------------------
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--manifest_out", default=None)

    # -------------------------------------------------
    # Pair construction
    # -------------------------------------------------
    parser.add_argument("--pair_column", required=True)

    parser.add_argument(
        "--pair_mode",
        choices=["sample", "all"],
        default="sample",
    )

    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--pair_start", type=int, default=0)
    parser.add_argument("--pair_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # -------------------------------------------------
    # Time projection
    # -------------------------------------------------
    parser.add_argument(
        "--time_mode",
        choices=[
            "point",
            "window",
            "full",
            "mean",
            "sliding",
        ],
        default="full",
    )

    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.6)

    parser.add_argument(
        "--target_time",
        type=float,
        default=None,
    )

    # -------------------------------------------------
    # Spatial selection
    # -------------------------------------------------
    parser.add_argument("--spatial_preset", default=None)

    parser.add_argument(
        "--spatial_method",
        choices=[
            "all",
            "prefix",
            "name",
            "regex",
            "index",
        ],
        default="all",
    )

    parser.add_argument("--spatial_values", default=None)

    # -------------------------------------------------
    # CV strategy
    # -------------------------------------------------
    parser.add_argument(
        "--cv",
        choices=[
            "loso",
            "leave_one_group_out",
            "leave_one_run_out",
            "leave_one_block_out",
            "stratified_kfold",
        ],
        default="loso",
    )

    parser.add_argument(
        "--group_column",
        default=None,
        help=(
            "Metadata column used for "
            "run/block/group CV."
        ),
    )

    parser.add_argument(
        "--subject_index",
        type=int,
        default=0,
        help=(
            "Subject index for "
            "trial-level CV decoding."
        ),
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )

    # -------------------------------------------------
    # Classifier
    # -------------------------------------------------
    parser.add_argument(
        "--classifier",
        default="linear_svm",
    )

    parser.add_argument(
        "--scaler",
        default="standard",
    )

    parser.add_argument("--C", type=float, default=1.0)

    parser.add_argument(
        "--max_iter",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--class_weight",
        default=None,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--metrics",
        default="accuracy,balanced_accuracy",
    )

    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.manifest is None and args.feature_dir is None:
        raise ValueError(
            "Provide either --manifest or --feature_dir."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load manifest
    # -------------------------------------------------
    if (
        args.manifest is not None
        and Path(args.manifest).exists()
    ):

        manifest = load_manifest(args.manifest)
        manifest_path = Path(args.manifest)

        print(f"[MANIFEST] Loaded: {manifest_path}")

    else:

        if args.feature_dir is None:
            raise ValueError(
                "Manifest not found and "
                "--feature_dir not provided."
            )

        if args.manifest_out is None:
            manifest_path = out_dir / "auto_manifest.csv"
        else:
            manifest_path = Path(args.manifest_out)

        print(
            "[MANIFEST] Building manifest "
            "automatically from feature_dir"
        )

        manifest = build_manifest_from_feature_dir(
            feature_dir=args.feature_dir,
            output_path=manifest_path,
        )

    validate_manifest_feature_consistency(
        manifest,
        require_same_feature_space=True,
        require_same_representation=True,
        require_same_task=False,
    )

    print("\n=====================================")
    print("General Pairwise Decoding")
    print("=====================================")
    print("n rows in manifest:", len(manifest))
    print("pair_column:", args.pair_column)
    print("pair_mode:", args.pair_mode)
    print("time_mode:", args.time_mode)
    print("cv:", args.cv)
    print("classifier:", args.classifier)
    print("out_dir:", out_dir)
    print("=====================================\n")

    # -------------------------------------------------
    # Load features
    # -------------------------------------------------
    feature_list, meta_list = load_features_from_manifest(
        manifest,
        mmap=args.mmap,
    )

    metadata_list = [
        metadata_from_meta(m)
        for m in meta_list
    ]

    # -------------------------------------------------
    # Pair table
    # -------------------------------------------------
    meta0_df = metadata_list[0]

    if args.pair_column not in meta0_df.columns:
        raise KeyError(
            f"pair_column '{args.pair_column}' "
            f"not found."
        )

    pair_table = build_pairs_from_column(
        metadata=meta0_df,
        column=args.pair_column,
        unique_only=True,
    )

    if args.pair_mode == "sample":

        pair_table = sample_pairs(
            pair_table=pair_table,
            n_pairs=args.n_pairs,
            seed=args.seed,
        )

    pair_table = chunk_pairs(
        pair_table=pair_table,
        pair_start=args.pair_start,
        pair_end=args.pair_end,
    )

    print("[PAIRS]")
    print("n pairs:", len(pair_table))
    print(pair_table.head())

    # -------------------------------------------------
    # Validate alignment
    # -------------------------------------------------
    validate_trial_alignment(
        metadata_list=metadata_list,
        column=args.pair_column,
        strict=True,
    )

    values_list = get_values_list(
        meta_list=meta_list,
        pair_column=args.pair_column,
    )

    # -------------------------------------------------
    # Spatial config
    # -------------------------------------------------
    if args.spatial_preset is not None:

        spatial_selection_map = build_ctf_roi_map()

        spatial_selection_name = args.spatial_preset

        spatial_method = "all"
        spatial_values = None

    else:

        spatial_selection_map = None
        spatial_selection_name = None

        spatial_method = args.spatial_method

        spatial_values = parse_spatial_values(
            args.spatial_values
        )

    # -------------------------------------------------
    # Projection
    # -------------------------------------------------
    X_list, projection_info_list = (
        project_multiple_subjects_to_X(
            feature_list=feature_list,
            meta_list=meta_list,
            time_mode=args.time_mode,
            tmin=args.tmin,
            tmax=args.tmax,
            target_time=args.target_time,
            spatial_method=spatial_method,
            spatial_values=spatial_values,
            spatial_selection_name=spatial_selection_name,
            spatial_selection_map=spatial_selection_map,
        )
    )

    print("[PROJECTION]")
    print("Example X shape:", X_list[0].shape)

    metrics = [
        m.strip()
        for m in args.metrics.split(",")
        if m.strip()
    ]

    # -------------------------------------------------
    # Run decoding
    # -------------------------------------------------
    if args.cv == "loso":

        pair_results_df, fold_results_df = (
            run_pairwise_loso_decoding(
                X_list=X_list,
                values_list=values_list,
                pair_table=pair_table,
                classifier=args.classifier,
                scaler=args.scaler,
                C=args.C,
                max_iter=args.max_iter,
                class_weight=args.class_weight,
                random_state=args.random_state,
                metrics=metrics,
                verbose=args.verbose,
            )
        )

    else:

        si = args.subject_index

        X = X_list[si]
        values = values_list[si]
        metadata_df = metadata_list[si]

        if args.cv == "stratified_kfold":

            splits = make_stratified_kfold_splits(
                y=values,
                n_splits=args.n_splits,
                shuffle=True,
                seed=args.seed,
            )

        else:

            if args.group_column is None:
                raise ValueError(
                    f"--group_column required "
                    f"for cv={args.cv}"
                )

            if args.group_column not in metadata_df.columns:
                raise KeyError(
                    f"group_column '{args.group_column}' "
                    f"not found."
                )

            group_values = (
                metadata_df[args.group_column]
                .astype(str)
                .to_numpy()
            )

            if args.cv == "leave_one_group_out":

                splits = (
                    make_leave_one_group_out_splits(
                        group_values
                    )
                )

            elif args.cv == "leave_one_run_out":

                splits = (
                    make_leave_one_run_out_splits(
                        group_values
                    )
                )

            elif args.cv == "leave_one_block_out":

                splits = (
                    make_leave_one_block_out_splits(
                        group_values
                    )
                )

            else:
                raise NotImplementedError(args.cv)

        pair_results_df, fold_results_df = (
            run_pairwise_single_subject_cv_decoding(
                X=X,
                values=values,
                pair_table=pair_table,
                splits=splits,
                classifier=args.classifier,
                scaler=args.scaler,
                C=args.C,
                max_iter=args.max_iter,
                class_weight=args.class_weight,
                random_state=args.random_state,
                metrics=metrics,
                verbose=args.verbose,
            )
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    feature_space = meta_list[0].get("feature_space")
    representation = meta_list[0].get("representation")
    task = meta_list[0].get("task")

    tag = (
        f"{task}"
        f"_{feature_space}-{representation}"
        f"_pair-{args.pair_column}"
        f"_time-{args.time_mode}"
        f"_cv-{args.cv}"
    )

    if args.spatial_preset is not None:
        tag += f"_space-{args.spatial_preset}"
    else:
        tag += f"_space-{args.spatial_method}"

    tag += (
        f"_{args.pair_mode}"
        f"_seed{args.seed}"
        f"_chunk{args.pair_start}-"
        f"{args.pair_end if args.pair_end is not None else 'end'}"
    )

    pair_csv = out_dir / f"{tag}_pair_results.csv"
    fold_csv = out_dir / f"{tag}_fold_results.csv"

    pair_table_csv = out_dir / f"{tag}_pair_table.csv"

    run_meta_json = out_dir / f"{tag}_run_meta.json"

    pair_results_df.to_csv(pair_csv, index=False)
    fold_results_df.to_csv(fold_csv, index=False)
    pair_table.to_csv(pair_table_csv, index=False)

    run_meta = {
        "script": "svm/scripts/run_pairwise_decoding.py",
        "manifest": str(manifest_path),
        "task": task,
        "feature_space": feature_space,
        "representation": representation,
        "pair_column": args.pair_column,
        "pair_mode": args.pair_mode,
        "cv": args.cv,
        "group_column": args.group_column,
        "subject_index": args.subject_index,
        "n_splits": args.n_splits,
        "time_mode": args.time_mode,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "target_time": args.target_time,
        "classifier": args.classifier,
        "metrics": metrics,
        "projection_example": projection_info_list[0],
    }

    save_json(run_meta_json, run_meta)

    print("\nSaved:")
    print(pair_csv)
    print(fold_csv)
    print(pair_table_csv)
    print(run_meta_json)

    print("\nDone.")


if __name__ == "__main__":
    main()
