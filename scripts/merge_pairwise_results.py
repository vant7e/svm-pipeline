# svm/scripts/merge_pairwise_results.py

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_files(results_dir, pattern):
    return sorted(Path(results_dir).glob(pattern))


def safe_concat(files):
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def summarize_pair_results(pair_df, accuracy_col="accuracy_mean"):
    summary = {}

    summary["n_rows"] = int(len(pair_df))

    if "status" in pair_df.columns:
        summary["n_success"] = int((pair_df["status"] == "ok").sum())
        summary["n_failed"] = int((pair_df["status"] == "failed").sum())
    else:
        summary["n_success"] = int(len(pair_df))
        summary["n_failed"] = 0

    if accuracy_col in pair_df.columns:
        acc = pd.to_numeric(pair_df[accuracy_col], errors="coerce").dropna()

        summary["accuracy_col"] = accuracy_col
        summary["n_accuracy_valid"] = int(len(acc))

        if len(acc) > 0:
            summary["mean_accuracy"] = float(acc.mean())
            summary["median_accuracy"] = float(acc.median())
            summary["std_accuracy"] = float(acc.std(ddof=1))
            summary["sem_accuracy"] = float(acc.std(ddof=1) / np.sqrt(len(acc)))
            summary["min_accuracy"] = float(acc.min())
            summary["max_accuracy"] = float(acc.max())
            summary["n_above_0.5"] = int((acc > 0.5).sum())
            summary["prop_above_0.5"] = float((acc > 0.5).mean())
        else:
            summary["mean_accuracy"] = None

    return summary


def plot_accuracy_hist(pair_df, out_path, accuracy_col="accuracy_mean"):
    if accuracy_col not in pair_df.columns:
        print(f"[WARN] {accuracy_col} not found; skipping histogram.")
        return

    acc = pd.to_numeric(pair_df[accuracy_col], errors="coerce").dropna()

    if len(acc) == 0:
        print("[WARN] No valid accuracy values; skipping histogram.")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(acc, bins=40)
    plt.axvline(0.5, linestyle="--", label="Chance = 0.5")
    plt.axvline(acc.mean(), linestyle="-", label=f"Mean = {acc.mean():.3f}")
    plt.xlabel("Pairwise decoding accuracy")
    plt.ylabel("Count")
    plt.title("Pairwise decoding accuracy distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_chunk_completion_table(results_dir):
    results_dir = Path(results_dir)

    done_files = sorted(results_dir.glob("chunk_*.done"))

    rows = []

    for f in done_files:
        stem = f.stem
        # expected: chunk_START_END
        parts = stem.split("_")

        if len(parts) >= 3:
            try:
                pair_start = int(parts[1])
                pair_end = int(parts[2])
            except Exception:
                pair_start = None
                pair_end = None
        else:
            pair_start = None
            pair_end = None

        rows.append({
            "done_file": str(f),
            "pair_start": pair_start,
            "pair_end": pair_end,
        })

    return pd.DataFrame(rows)


def collect_run_metadata(results_dir):
    meta_files = find_files(results_dir, "*_run_meta.json")

    metas = []

    for f in meta_files:
        try:
            meta = load_json(f)
            meta["source_meta_file"] = str(f)
            metas.append(meta)
        except Exception as e:
            print(f"[WARN] Could not read meta {f}: {e}")

    if len(metas) == 0:
        return {}

    first = metas[0]

    keep_keys = [
        "task",
        "feature_space",
        "representation",
        "pair_column",
        "pair_mode",
        "time_mode",
        "tmin",
        "tmax",
        "target_time",
        "spatial_preset",
        "spatial_method",
        "spatial_values",
        "classifier",
        "scaler",
        "C",
        "max_iter",
        "class_weight",
        "random_state",
        "metrics",
    ]

    out = {}

    for k in keep_keys:
        if k in first:
            out[k] = first[k]

    out["n_run_meta_files"] = len(meta_files)

    return out


def cleanup_chunk_files(results_dir, dry_run=False):
    """
    Remove chunk-level files after successful merge.

    Deletes:
        *_pair_results.csv
        *_fold_results.csv
        *_pair_table.csv
        *_run_meta.json
        chunk_*.done

    Does not delete:
        merged/
        logs
    """

    results_dir = Path(results_dir)

    patterns = [
        "*_pair_results.csv",
        "*_fold_results.csv",
        "*_pair_table.csv",
        "*_run_meta.json",
        "chunk_*.done",
    ]

    files = []

    for p in patterns:
        files.extend(sorted(results_dir.glob(p)))

    deleted = []

    for f in files:
        if dry_run:
            print(f"[DRY RUN] Would delete: {f}")
        else:
            f.unlink()
            deleted.append(str(f))

    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Merge chunked pairwise decoding results."
    )

    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", default=None)

    parser.add_argument(
        "--accuracy_col",
        default="accuracy_mean",
        help="Column to summarize/plot, default from aggregate_fold_results.",
    )

    parser.add_argument("--cleanup_chunks", action="store_true")
    parser.add_argument("--dry_run_cleanup", action="store_true")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.out_dir is None:
        out_dir = results_dir / "merged"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=====================================")
    print("Merging pairwise results")
    print("=====================================")
    print("results_dir:", results_dir)
    print("out_dir:", out_dir)
    print("accuracy_col:", args.accuracy_col)
    print("cleanup_chunks:", args.cleanup_chunks)
    print("=====================================\n")

    pair_files = find_files(results_dir, "*_pair_results.csv")
    fold_files = find_files(results_dir, "*_fold_results.csv")
    pair_table_files = find_files(results_dir, "*_pair_table.csv")

    print(f"pair result files: {len(pair_files)}")
    print(f"fold result files: {len(fold_files)}")
    print(f"pair table files: {len(pair_table_files)}")

    pair_df = safe_concat(pair_files)
    fold_df = safe_concat(fold_files)
    pair_table_df = safe_concat(pair_table_files)

    if pair_df.empty:
        raise ValueError("No pair result files found or readable.")

    # sort if possible
    sort_cols = []
    for c in ["pair_id", "pair_index", "value_a", "value_b"]:
        if c in pair_df.columns:
            sort_cols.append(c)

    if sort_cols:
        pair_df = pair_df.sort_values(sort_cols).reset_index(drop=True)

    if not fold_df.empty:
        fold_sort_cols = []
        for c in ["pair_id", "pair_index", "split_id"]:
            if c in fold_df.columns:
                fold_sort_cols.append(c)

        if fold_sort_cols:
            fold_df = fold_df.sort_values(fold_sort_cols).reset_index(drop=True)

    # save merged csvs
    merged_pair_csv = out_dir / "merged_pair_results.csv"
    merged_fold_csv = out_dir / "merged_fold_results.csv"
    merged_pair_table_csv = out_dir / "merged_pair_table.csv"

    pair_df.to_csv(merged_pair_csv, index=False)

    if not fold_df.empty:
        fold_df.to_csv(merged_fold_csv, index=False)

    if not pair_table_df.empty:
        pair_table_df.to_csv(merged_pair_table_csv, index=False)

    # top/bottom
    if args.accuracy_col in pair_df.columns:
        acc_numeric = pd.to_numeric(pair_df[args.accuracy_col], errors="coerce")
        pair_df["_acc_numeric"] = acc_numeric

        pair_df.sort_values("_acc_numeric", ascending=False).head(20).drop(
            columns=["_acc_numeric"]
        ).to_csv(out_dir / "top20_pairs.csv", index=False)

        pair_df.sort_values("_acc_numeric", ascending=True).head(20).drop(
            columns=["_acc_numeric"]
        ).to_csv(out_dir / "bottom20_pairs.csv", index=False)

        pair_df = pair_df.drop(columns=["_acc_numeric"])

    # plot
    hist_path = out_dir / "accuracy_hist.png"
    plot_accuracy_hist(
        pair_df=pair_df,
        out_path=hist_path,
        accuracy_col=args.accuracy_col,
    )

    # chunk table
    chunk_df = build_chunk_completion_table(results_dir)
    chunk_csv = out_dir / "chunk_completion.csv"
    chunk_df.to_csv(chunk_csv, index=False)

    # metadata
    pipeline_meta = collect_run_metadata(results_dir)

    summary = summarize_pair_results(
        pair_df=pair_df,
        accuracy_col=args.accuracy_col,
    )

    summary.update({
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "n_pair_files": len(pair_files),
        "n_fold_files": len(fold_files),
        "n_pair_table_files": len(pair_table_files),
        "n_done_files": int(len(chunk_df)),
        "merged_pair_results": str(merged_pair_csv),
        "merged_fold_results": str(merged_fold_csv) if not fold_df.empty else None,
        "merged_pair_table": str(merged_pair_table_csv) if not pair_table_df.empty else None,
        "accuracy_hist": str(hist_path),
        "chunk_completion": str(chunk_csv),
        "pipeline": pipeline_meta,
    })

    summary_json = out_dir / "merge_summary.json"
    save_json(summary_json, summary)

    summary_txt = out_dir / "merge_summary.txt"
    with open(summary_txt, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved:")
    print(merged_pair_csv)
    if not fold_df.empty:
        print(merged_fold_csv)
    if not pair_table_df.empty:
        print(merged_pair_table_csv)
    print(summary_json)
    print(summary_txt)
    print(hist_path)
    print(chunk_csv)

    # cleanup
    if args.cleanup_chunks:
        print("\n[CLEANUP]")
        deleted = cleanup_chunk_files(
            results_dir=results_dir,
            dry_run=args.dry_run_cleanup,
        )

        if args.dry_run_cleanup:
            print("Dry-run cleanup complete.")
        else:
            print(f"Deleted {len(deleted)} chunk-level files.")

        cleanup_log = out_dir / "cleanup_deleted_files.json"
        save_json(cleanup_log, {
            "dry_run": args.dry_run_cleanup,
            "deleted_files": deleted,
        })

        print(cleanup_log)

    print("\nDone.")


if __name__ == "__main__":
    main()