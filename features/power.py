# svm/features/power.py

import numpy as np
from pathlib import Path
from mne.time_frequency import tfr_morlet

from svm.core.schema import build_feature_meta
from svm.core.io import (
    save_feature_package,
    load_feature_package,
    should_compute,
    build_feature_paths,
)


VALID_POWER_REPRESENTATIONS = {
    "magnitude",
    "magnitude_squared",
    "log_power",
}


def parse_freqs(freq_string):
    """
    Parse frequency string.

    Examples
    --------
    "2:30:1" -> array([2, 3, ..., 30])
    "4,8,12" -> array([4, 8, 12])
    """

    if ":" in freq_string:
        start, stop, step = map(float, freq_string.split(":"))
        return np.arange(start, stop + step, step)

    return np.array([float(f) for f in freq_string.split(",")])


def crop_epochs_if_needed(epochs, time_range=None):
    """
    Crop epochs before time-frequency decomposition.

    time_range example:
        (0.0, 0.6)
    """

    if time_range is None:
        return epochs

    tmin, tmax = time_range
    return epochs.copy().crop(tmin=tmin, tmax=tmax)


def compute_power_feature(
    epochs,
    freq_str="2:30:1",
    representation="magnitude_squared",
    n_jobs=1,
    chunk_size=5,
    time_range=None,
):
    """
    Compute power feature from MNE Epochs.

    Standard output shape:
        (freq, time, trial, channel)

    Representations
    ---------------
    magnitude:
        abs(complex TFR)

    magnitude_squared:
        abs(complex TFR) ** 2

    log_power:
        log1p(abs(complex TFR) ** 2)

    Notes
    -----
    This file only computes pure power/magnitude representations.
    Any later frequency-band averaging, time-window averaging, RSA distance,
    or classifier projection should happen downstream.
    """

    if representation not in VALID_POWER_REPRESENTATIONS:
        raise ValueError(
            f"Invalid power representation: {representation}. "
            f"Options: {sorted(VALID_POWER_REPRESENTATIONS)}"
        )

    epochs = crop_epochs_if_needed(
        epochs,
        time_range=time_range,
    )

    freqs = parse_freqs(freq_str)
    n_cycles = freqs / 2.0

    power_chunks = []

    for i in range(0, len(freqs), chunk_size):
        f_chunk = freqs[i:i + chunk_size]
        n_chunk = n_cycles[i:i + chunk_size]

        tfr = tfr_morlet(
            epochs,
            freqs=f_chunk,
            n_cycles=n_chunk,
            return_itc=False,
            output="complex",
            average=False,
            use_fft=True,
            n_jobs=n_jobs,
            verbose=False,
        )

        # MNE output:
        # (trial, channel, freq_chunk, time)
        magnitude = np.abs(tfr.data)

        if representation == "magnitude":
            power_chunk = magnitude

        elif representation == "magnitude_squared":
            power_chunk = magnitude ** 2

        elif representation == "log_power":
            power_chunk = np.log1p(magnitude ** 2)

        else:
            raise RuntimeError(f"Unhandled representation: {representation}")

        power_chunks.append(power_chunk)

    power = np.concatenate(power_chunks, axis=2)

    # power:
    # (trial, channel, freq, time)
    # -> standard:
    # (freq, time, trial, channel)
    feature = np.transpose(power, (2, 3, 0, 1)).astype(np.float32)

    times = tfr.times

    return feature, freqs.astype(float), times


def compute_and_save_power_feature(
    epochs,
    out_dir,
    subject,
    task,
    representation="magnitude_squared",
    freq_str="2:30:1",
    n_jobs=1,
    chunk_size=5,
    time_range=None,
    trial_ids=None,
    labels=None,
    source_epoch_file=None,
    suffix=None,
    overwrite=False,
):
    """
    Compute or load power feature package.

    Saves
    -----
    feature:
        .npy file

    metadata:
        .meta.json file

    Standard feature shape:
        (freq, time, trial, channel)
    """

    if representation not in VALID_POWER_REPRESENTATIONS:
        raise ValueError(
            f"Invalid power representation: {representation}. "
            f"Options: {sorted(VALID_POWER_REPRESENTATIONS)}"
        )

    feature_path, meta_path = build_feature_paths(
        out_dir=out_dir,
        subject=subject,
        task=task,
        feature_space="power",
        representation=representation,
        suffix=suffix,
    )

    if not should_compute(feature_path, overwrite=overwrite):
        print(f"[SKIP] Power feature exists: {feature_path}")
        return load_feature_package(
            feature_path=feature_path,
            meta_path=meta_path,
            mmap=True,
        )

    print("[POWER] Computing power feature")
    print("Subject:", subject)
    print("Task:", task)
    print("Representation:", representation)
    print("Frequency string:", freq_str)
    print("Time range:", time_range)

    feature, freqs, times = compute_power_feature(
        epochs=epochs,
        freq_str=freq_str,
        representation=representation,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        time_range=time_range,
    )

    notes = (
        f"Power feature representation: {representation}. "
        f"Computed using Morlet wavelets with n_cycles=freq/2. "
        f"Frequency/time projection should be handled downstream."
    )

    meta = build_feature_meta(
        subject=subject,
        task=task,
        feature_space="power",
        representation=representation,
        feature=feature,
        freqs=freqs,
        times=times,
        ch_names=epochs.ch_names,
        trial_ids=trial_ids,
        labels=labels,
        source_epoch_file=source_epoch_file,
        notes=notes,
    )

    meta["freq_str"] = freq_str
    meta["n_jobs"] = n_jobs
    meta["chunk_size"] = chunk_size
    meta["time_range"] = list(time_range) if time_range is not None else None

    save_feature_package(
        feature_path=feature_path,
        meta_path=meta_path,
        feature=feature,
        meta=meta,
    )

    print(f"[POWER] Saved feature: {feature_path}")
    print(f"[POWER] Saved metadata: {meta_path}")

    return feature, meta


def load_power_feature(
    feature_path,
    meta_path=None,
    mmap=True,
):
    """
    Load power feature package.

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

    if meta["feature_space"] != "power":
        raise ValueError(
            f"Expected feature_space='power', got {meta['feature_space']}"
        )

    return feature, meta