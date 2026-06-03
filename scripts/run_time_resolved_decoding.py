# svm/scripts/run_time_resolved_decoding.py

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

from svm.decode.projection import project_multiple_subjects_to_X

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


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def parse_spatial_values(values):
    if values is None or values.strip() == "":
        return None
    return [v.strip() for v in values.split(",") if v.strip()]


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
        raise KeyError("Feature metadata does not contain labels.")
    return pd.DataFrame(meta["labels"])


def get_values_list(meta_list, pair_column):
    values_list = []

    for i, meta in enumerate(meta_list):
        if "labels" not in meta or meta["labels"] is None:
            raise KeyError(f"Metadata index {i} has no labels.")

        if pair_column not in meta["labels"]:
            raise KeyError(
                f"pair_column '{pair_column}' not found in metadata labels. "
                f"Available columns: {list(meta['labels'].keys())}"
            )

        values_list.append(np.asarray(meta["labels"][pair_column]).astype(str))

    return values_list


def build_time_windows(times, tmin, tmax, window_ms, step_ms):
    times = np.asarray(times, dtype=float)

    if tmin is None:
        tmin = float(times[0])

    if tmax is None:
        tmax = float(times[-1])

    win = window_ms / 1000.0
    step = step_ms / 1000.0

    starts = np.arange(tmin, tmax - win + 1e-12, step)

    windows = []

    for i, start in enumerate(starts):
        end = start + win
        idx = np.where((times >= start) & (times <= end))[0]

        if len(idx) == 0:
            continue

        windows.append({
            "window_id": i,
            "t_start": float(start),
            "t_end": float(end),
            "t_center": float((start + end) / 2.0),
            "n_timepoints": int(len(idx)),
        })

    if len(windows) == 0:
        raise ValueError(
            f"No valid time windows found. "
            f"tmin={tmin}, tmax={tmax}, window_ms={window_ms}, step_ms={step_ms}"
        )

    return windows


def build_trial_level_splits(args, metadata_df, values):
    if args.cv == "stratified_kfold":
        return make_stratified_kfold_splits(
            y=values,
            n_splits=args.n_splits,
            shuffle=True,
            seed=args.seed,
        )

    if args.group_column is None:
        raise ValueError(f"--group_column is required for cv={args.cv}")

    if args.group_column not in metadata_df.columns:
        raise KeyError(
            f"group_column '{args.group_column}' not found. "
            f"Available columns: {list(metadata_df.columns)}"
        )

    group_values = metadata_df[args.group_column].astype(str).to_numpy()

    if args.cv == "leave_one_group_out":
        return make_leave_one_group_out_splits(group_values)

    if args.cv == "leave_one_run_out":
        return make_leave_one_run_out_splits(group_values)

    if args.cv == "leave_one_block_out":
        return make_leave_one_block_out_splits(group_values)

    raise NotImplementedError(f"CV not implemented: {args.cv}")


def run_decoding_for_window(
    args,
    X_list,
    values_list,
    metadata_list,
    pair_table,
    metrics,
):
    if args.cv == "loso":
        return run_pairwise_loso_decoding(
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

    si = args.subject_index

    if si < 0 or si >= len(X_list):
        raise IndexError(
            f"subject_index={si} out of range for {len(X_list)} subjects."
        )

    X = X_list[si]
    values = values_list[si]
    metadata_df = metadata_list[si]

    splits = build_trial_level_splits(
        args=args,
        metadata_df=metadata_df,
        values=values,
    )

    return run_pairwise_single_subject_cv_decoding(
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


def main():
    parser = argparse.ArgumentParser(
        description="General time-resolved pairwise decoding framework."
    )

    # -------------------------
    # Dataset / manifest
    # -------------------------
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--manifest_out", default=None)

    # -------------------------
    # Pair construction
    # -------------------------
    parser.add_argument("--pair_column", required=True)
    parser.add_argument("--pair_mode", choices=["sample", "all"], default="sample")
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--pair_start", type=int, default=0)
    parser.add_argument("--pair_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # -------------------------
    # Time-resolved settings
    # -------------------------
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--window_ms", type=float, default=10.0)
    parser.add_argument("--step_ms", type=float, default=10.0)

    parser.add_argument(
        "--window_mode",
        choices=["window", "mean"],
        default="window",
        help="window = flatten all timepoints; mean = average within each window.",
    )

    parser.add_argument("--window_start", type=int, default=0)
    parser.add_argument("--window_end", type=int, default=None)

    # -------------------------
    # Spatial selection
    # -------------------------
    parser.add_argument("--spatial_preset", default=None)
    parser.add_argument(
        "--spatial_method",
        choices=["all", "prefix", "name", "regex", "index"],
        default="all",
    )
    parser.add_argument("--spatial_values", default=None)

    # -------------------------
    # CV strategy
    # -------------------------
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
    parser.add_argument("--group_column", default=None)
    parser.add_argument("--subject_index", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)

    # -------------------------
    # Classifier
    # -------------------------
    parser.add_argument("--classifier", default="linear_svm")
    parser.add_argument("--scaler", default="standard")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--random_state", type=int, default=42)

    # -------------------------
    # Output
    # -------------------------
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--metrics", default="accuracy,balanced_accuracy")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    if args.manifest is None and args.feature_dir is None:
        raise ValueError("Provide either --manifest or --feature_dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load or build manifest
    # -------------------------
    if args.manifest is not None and Path(args.manifest).exists():
        manifest = load_manifest(args.manifest)
        manifest_path = Path(args.manifest)
        print(f"[MANIFEST] Loaded: {manifest_path}")

    else:
        if args.feature_dir is None:
            raise ValueError(
                "Manifest not found and --feature_dir was not provided."
            )

        manifest_path = Path(args.manifest_out) if args.manifest_out else out_dir / "auto_manifest.csv"

        print("[MANIFEST] Building manifest automatically from feature_dir")
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

    # -------------------------
    # Load features
    # -------------------------
    feature_list, meta_list = load_features_from_manifest(
        manifest,
        mmap=args.mmap,
    )

    metadata_list = [metadata_from_meta(m) for m in meta_list]

    feature_space = meta_list[0].get("feature_space")
    representation = meta_list[0].get("representation")
    task = meta_list[0].get("task")
    times = np.asarray(meta_list[0]["times"], dtype=float)

    # -------------------------
    # Pair table
    # -------------------------
    meta0_df = metadata_list[0]

    if args.pair_column not in meta0_df.columns:
        raise KeyError(
            f"pair_column '{args.pair_column}' not found. "
            f"Available columns: {list(meta0_df.columns)}"
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

    validate_trial_alignment(
        metadata_list=metadata_list,
        column=args.pair_column,
        strict=True,
    )

    values_list = get_values_list(
        meta_list=meta_list,
        pair_column=args.pair_column,
    )

    # -------------------------
    # Spatial config
    # -------------------------
    if args.spatial_preset is not None:
        spatial_selection_map = build_ctf_roi_map()
        spatial_selection_name = args.spatial_preset
        spatial_method = "all"
        spatial_values = None
    else:
        spatial_selection_map = None
        spatial_selection_name = None
        spatial_method = args.spatial_method
        spatial_values = parse_spatial_values(args.spatial_values)

    # -------------------------
    # Build time windows
    # -------------------------
    windows = build_time_windows(
        times=times,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
    )

    total_windows = len(windows)

    if args.window_end is None:
        args.window_end = total_windows

    selected_windows = windows[args.window_start:args.window_end]

    if len(selected_windows) == 0:
        raise ValueError("No windows selected.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("\n=====================================")
    print("Time-resolved Pairwise Decoding")
    print("=====================================")
    print("n manifest rows:", len(manifest))
    print("feature:", feature_space, representation)
    print("pair_column:", args.pair_column)
    print("pair_mode:", args.pair_mode)
    print("n pairs:", len(pair_table))
    print("cv:", args.cv)
    print("classifier:", args.classifier)
    print("metrics:", metrics)
    print("window_mode:", args.window_mode)
    print("total windows:", total_windows)
    print("selected windows:", args.window_start, args.window_end)
    print("out_dir:", out_dir)
    print("=====================================\n")

    all_window_summaries = []

    # -------------------------
    # Time loop
    # -------------------------
    for w in selected_windows:
        wid = w["window_id"]
        t_start = w["t_start"]
        t_end = w["t_end"]

        tag = (
            f"{task}"
            f"_{feature_space}-{representation}"
            f"_pair-{args.pair_column}"
            f"_timeresolved"
            f"_w{wid:04d}"
            f"_t{t_start:.4f}-{t_end:.4f}"
            f"_cv-{args.cv}"
        )

        if args.spatial_preset is not None:
            tag += f"_space-{args.spatial_preset}"
        else:
            tag += f"_space-{args.spatial_method}"

        tag += (
            f"_{args.pair_mode}"
            f"_seed{args.seed}"
            f"_pairs{args.pair_start}-"
            f"{args.pair_end if args.pair_end is not None else 'end'}"
        )

        pair_csv = out_dir / f"{tag}_pair_results.csv"
        fold_csv = out_dir / f"{tag}_fold_results.csv"
        run_meta_json = out_dir / f"{tag}_run_meta.json"

        if args.skip_existing and pair_csv.exists() and fold_csv.exists():
            print(f"[SKIP] Window {wid}: existing outputs found.")
            all_window_summaries.append({
                "window_id": wid,
                "t_start": t_start,
                "t_end": t_end,
                "t_center": w["t_center"],
                "n_timepoints": w["n_timepoints"],
                "pair_csv": str(pair_csv),
                "fold_csv": str(fold_csv),
                "run_meta_json": str(run_meta_json),
                "status": "skipped_existing",
            })
            continue

        print(
            f"\n[WINDOW {wid}/{total_windows}] "
            f"{t_start:.4f}–{t_end:.4f}s "
            f"({w['n_timepoints']} timepoints)",
            flush=True,
        )

        X_list, projection_info_list = project_multiple_subjects_to_X(
            feature_list=feature_list,
            meta_list=meta_list,
            time_mode=args.window_mode,
            tmin=t_start,
            tmax=t_end,
            spatial_method=spatial_method,
            spatial_values=spatial_values,
            spatial_selection_name=spatial_selection_name,
            spatial_selection_map=spatial_selection_map,
        )

        pair_results_df, fold_results_df = run_decoding_for_window(
            args=args,
            X_list=X_list,
            values_list=values_list,
            metadata_list=metadata_list,
            pair_table=pair_table,
            metrics=metrics,
        )

        for df in [pair_results_df, fold_results_df]:
            df["window_id"] = wid
            df["t_start"] = t_start
            df["t_end"] = t_end
            df["t_center"] = w["t_center"]
            df["n_timepoints"] = w["n_timepoints"]

        pair_results_df.to_csv(pair_csv, index=False)
        fold_results_df.to_csv(fold_csv, index=False)

        run_meta = {
            "script": "svm/scripts/run_time_resolved_decoding.py",
            "manifest": str(manifest_path),
            "task": task,
            "feature_space": feature_space,
            "representation": representation,
            "pair_column": args.pair_column,
            "pair_mode": args.pair_mode,
            "n_pairs": int(len(pair_table)),
            "cv": args.cv,
            "group_column": args.group_column,
            "subject_index": args.subject_index,
            "n_splits": args.n_splits,
            "window_id": wid,
            "t_start": t_start,
            "t_end": t_end,
            "t_center": w["t_center"],
            "n_timepoints": w["n_timepoints"],
            "window_mode": args.window_mode,
            "window_ms": args.window_ms,
            "step_ms": args.step_ms,
            "spatial_preset": args.spatial_preset,
            "spatial_method": args.spatial_method,
            "spatial_values": args.spatial_values,
            "classifier": args.classifier,
            "scaler": args.scaler,
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "random_state": args.random_state,
            "metrics": metrics,
            "projection_example": projection_info_list[0],
            "pair_csv": str(pair_csv),
            "fold_csv": str(fold_csv),
        }

        save_json(run_meta_json, run_meta)

        summary_row = {
            "window_id": wid,
            "t_start": t_start,
            "t_end": t_end,
            "t_center": w["t_center"],
            "n_timepoints": w["n_timepoints"],
            "pair_csv": str(pair_csv),
            "fold_csv": str(fold_csv),
            "run_meta_json": str(run_meta_json),
            "status": "ok",
        }

        if "accuracy_mean" in pair_results_df.columns:
            acc = pd.to_numeric(pair_results_df["accuracy_mean"], errors="coerce")
            summary_row["mean_accuracy_across_pairs"] = float(acc.mean())
            summary_row["median_accuracy_across_pairs"] = float(acc.median())
            summary_row["std_accuracy_across_pairs"] = float(acc.std(ddof=1))
            summary_row["n_pairs_valid"] = int(acc.notna().sum())

        all_window_summaries.append(summary_row)

    # -------------------------
    # Save global summary
    # -------------------------
    time_summary = pd.DataFrame(all_window_summaries)
    time_summary_csv = out_dir / "time_summary.csv"
    time_summary.to_csv(time_summary_csv, index=False)

    global_meta = {
        "script": "svm/scripts/run_time_resolved_decoding.py",
        "manifest": str(manifest_path),
        "feature_space": feature_space,
        "representation": representation,
        "task": task,
        "pair_column": args.pair_column,
        "pair_mode": args.pair_mode,
        "n_pairs": int(len(pair_table)),
        "cv": args.cv,
        "group_column": args.group_column,
        "subject_index": args.subject_index,
        "n_splits": args.n_splits,
        "n_total_windows": total_windows,
        "n_selected_windows": len(selected_windows),
        "window_start": args.window_start,
        "window_end": args.window_end,
        "window_mode": args.window_mode,
        "window_ms": args.window_ms,
        "step_ms": args.step_ms,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "spatial_preset": args.spatial_preset,
        "spatial_method": args.spatial_method,
        "spatial_values": args.spatial_values,
        "classifier": args.classifier,
        "scaler": args.scaler,
        "C": args.C,
        "max_iter": args.max_iter,
        "class_weight": args.class_weight,
        "random_state": args.random_state,
        "metrics": metrics,
        "time_summary_csv": str(time_summary_csv),
    }

    global_meta_json = out_dir / "time_resolved_run_meta.json"
    save_json(global_meta_json, global_meta)

    print("\nSaved:")
    print(time_summary_csv)
    print(global_meta_json)
    print("\nDone.")


if __name__ == "__main__":
    main()
