# svm/scripts/extract_features.py

import argparse
from pathlib import Path

import mne
import pandas as pd

from svm.features.raw import compute_and_save_raw_feature
from svm.features.power import compute_and_save_power_feature
from svm.features.phase import compute_and_save_phase_feature
from svm.core.io import print_feature_summary


def parse_time_range(tmin, tmax):
    """
    Convert tmin/tmax arguments into time_range tuple.
    """

    if tmin is None and tmax is None:
        return None

    if tmin is None or tmax is None:
        raise ValueError("Both --tmin and --tmax must be provided, or neither.")

    return (float(tmin), float(tmax))


def build_trial_ids(metadata):
    """
    Build generic trial IDs from metadata row order.
    """

    return metadata.index.astype(str).tolist()


def build_labels_from_metadata(metadata):
    """
    Store all metadata columns as labels.

    This keeps the framework task-general:
        image
        OLD_NEW
        Confidence
        RT
        run
        category
        etc.
    """

    labels = {}

    for col in metadata.columns:
        labels[col] = metadata[col].astype(str).tolist()

    return labels


def extract_one_subject(
    epoch_path,
    metadata_path,
    out_dir,
    subject,
    task,
    feature_space,
    representation,
    freq_str="2:30:1",
    n_jobs=1,
    chunk_size=5,
    time_range=None,
    suffix=None,
    overwrite=False,
):
    """
    Extract one feature package for one subject/session.
    """

    epoch_path = Path(epoch_path)
    metadata_path = Path(metadata_path)
    out_dir = Path(out_dir)

    if not epoch_path.exists():
        raise FileNotFoundError(epoch_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    print("\n=====================================")
    print("Extracting feature")
    print("=====================================")
    print("Subject:", subject)
    print("Task:", task)
    print("Feature space:", feature_space)
    print("Representation:", representation)
    print("Epoch file:", epoch_path)
    print("Metadata file:", metadata_path)
    print("Output dir:", out_dir)
    print("Time range:", time_range)
    print("=====================================\n")

    epochs = mne.read_epochs(
        epoch_path,
        preload=True,
        verbose=False,
    )

    metadata = pd.read_csv(metadata_path)

    if len(metadata) != len(epochs):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) do not match "
            f"epochs ({len(epochs)})."
        )

    trial_ids = build_trial_ids(metadata)
    labels = build_labels_from_metadata(metadata)

    source_epoch_file = str(epoch_path)

    if feature_space == "raw":
        feature, meta = compute_and_save_raw_feature(
            epochs=epochs,
            out_dir=out_dir,
            subject=subject,
            task=task,
            time_range=time_range,
            trial_ids=trial_ids,
            labels=labels,
            source_epoch_file=source_epoch_file,
            suffix=suffix,
            overwrite=overwrite,
        )

    elif feature_space == "power":
        feature, meta = compute_and_save_power_feature(
            epochs=epochs,
            out_dir=out_dir,
            subject=subject,
            task=task,
            representation=representation,
            freq_str=freq_str,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            time_range=time_range,
            trial_ids=trial_ids,
            labels=labels,
            source_epoch_file=source_epoch_file,
            suffix=suffix,
            overwrite=overwrite,
        )

    elif feature_space == "phase":
        feature, meta = compute_and_save_phase_feature(
            epochs=epochs,
            out_dir=out_dir,
            subject=subject,
            task=task,
            representation=representation,
            freq_str=freq_str,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            time_range=time_range,
            trial_ids=trial_ids,
            labels=labels,
            source_epoch_file=source_epoch_file,
            suffix=suffix,
            overwrite=overwrite,
        )

    else:
        raise ValueError(
            "feature_space must be one of: raw, power, phase"
        )

    print_feature_summary(meta)

    return feature, meta


def main():
    parser = argparse.ArgumentParser(
        description="Extract standardized neural features for general SVM decoding."
    )

    parser.add_argument("--epoch_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--subject", required=True)
    parser.add_argument("--task", required=True)

    parser.add_argument(
        "--feature_space",
        required=True,
        choices=["raw", "power", "phase"],
    )

    parser.add_argument(
        "--representation",
        default=None,
        help=(
            "raw: amplitude; "
            "power: magnitude / magnitude_squared / log_power; "
            "phase: angle / sin_cos / complex_unit"
        ),
    )

    parser.add_argument("--freq_str", default="2:30:1")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=5)

    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)

    parser.add_argument("--suffix", default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.feature_space == "raw":
        representation = "amplitude"
    else:
        if args.representation is None:
            if args.feature_space == "power":
                representation = "magnitude_squared"
            elif args.feature_space == "phase":
                representation = "angle"
        else:
            representation = args.representation

    time_range = parse_time_range(args.tmin, args.tmax)

    extract_one_subject(
        epoch_path=args.epoch_path,
        metadata_path=args.metadata_path,
        out_dir=args.out_dir,
        subject=args.subject,
        task=args.task,
        feature_space=args.feature_space,
        representation=representation,
        freq_str=args.freq_str,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        time_range=time_range,
        suffix=args.suffix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()