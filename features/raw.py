# svm/features/raw.py

import numpy as np
from pathlib import Path

from svm.core.schema import build_feature_meta
from svm.core.io import (
    save_feature_package,
    load_feature_package,
    should_compute,
    build_feature_paths,
)


def crop_epochs_if_needed(epochs, time_range=None):
    """
    Crop epochs before extracting raw amplitude.

    time_range example:
        (0.0, 0.6)
    """

    if time_range is None:
        return epochs

    tmin, tmax = time_range
    return epochs.copy().crop(tmin=tmin, tmax=tmax)


def compute_raw_feature(
    epochs,
    time_range=None,
):
    """
    Compute raw amplitude feature from MNE Epochs.

    MNE epochs data shape:
        (trial, channel, time)

    Standard feature output shape:
        (freq, time, trial, channel)

    For raw amplitude:
        freq dimension is fake and equals 1.

    Output:
        feature shape = (1, time, trial, channel)
    """

    epochs = crop_epochs_if_needed(
        epochs,
        time_range=time_range,
    )

    data = epochs.get_data()
    # (trial, channel, time)

    feature = np.transpose(data, (2, 0, 1))
    # (time, trial, channel)

    feature = feature[np.newaxis, ...]
    # (1, time, trial, channel)

    freqs = np.array([0.0], dtype=float)
    times = epochs.times

    return feature.astype(np.float32), freqs, times


def compute_and_save_raw_feature(
    epochs,
    out_dir,
    subject,
    task,
    time_range=None,
    trial_ids=None,
    labels=None,
    source_epoch_file=None,
    suffix=None,
    overwrite=False,
):
    """
    Compute or load raw amplitude feature package.

    Saves:
        feature .npy
        metadata .meta.json

    Standard feature shape:
        (1, time, trial, channel)
    """

    feature_path, meta_path = build_feature_paths(
        out_dir=out_dir,
        subject=subject,
        task=task,
        feature_space="raw",
        representation="amplitude",
        suffix=suffix,
    )

    if not should_compute(feature_path, overwrite=overwrite):
        print(f"[SKIP] Raw feature exists: {feature_path}")
        return load_feature_package(
            feature_path=feature_path,
            meta_path=meta_path,
            mmap=True,
        )

    print("[RAW] Computing raw amplitude feature")
    print("Subject:", subject)
    print("Task:", task)
    print("Time range:", time_range)

    feature, freqs, times = compute_raw_feature(
        epochs=epochs,
        time_range=time_range,
    )

    notes = (
        "Raw time-domain amplitude feature. "
        "This is preprocessed epoched sensor-level data, not unprocessed continuous raw recording. "
        "A fake frequency axis of length 1 is added for compatibility with power and phase features."
    )

    meta = build_feature_meta(
        subject=subject,
        task=task,
        feature_space="raw",
        representation="amplitude",
        feature=feature,
        freqs=freqs,
        times=times,
        ch_names=epochs.ch_names,
        trial_ids=trial_ids,
        labels=labels,
        source_epoch_file=source_epoch_file,
        notes=notes,
    )

    meta["time_range"] = list(time_range) if time_range is not None else None

    save_feature_package(
        feature_path=feature_path,
        meta_path=meta_path,
        feature=feature,
        meta=meta,
    )

    print(f"[RAW] Saved feature: {feature_path}")
    print(f"[RAW] Saved metadata: {meta_path}")

    return feature, meta


def load_raw_feature(
    feature_path,
    meta_path=None,
    mmap=True,
):
    """
    Load raw feature package.

    If meta_path is None, assumes:
        feature.npy
        feature.meta.json
    """

    feature_path = Path(feature_path)

    if meta_path is None:
        meta_path = feature_path.with_suffix(".meta.json")

    feature, meta = load_feature_package(
        feature_path=feature_path,
        meta_path=meta_path,
        mmap=mmap,
    )

    if meta["feature_space"] != "raw":
        raise ValueError(
            f"Expected feature_space='raw', got {meta['feature_space']}"
        )

    return feature, meta