# svm/features/phase.py

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


VALID_PHASE_REPRESENTATIONS = {
    "angle",
    "sin_cos",
    "complex_unit",
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


def crop_feature_time(feature, times, time_range=None):
    """
    Crop feature after TFR computation.

    Input feature shape:
        (freq, time, trial, channel)
    """

    if time_range is None:
        return feature, times

    tmin, tmax = time_range

    tidx = np.where((times >= tmin) & (times <= tmax))[0]

    if len(tidx) == 0:
        raise ValueError(
            f"No TFR time points found between {tmin} and {tmax}. "
            f"Available range: {times[0]} to {times[-1]}"
        )

    feature = feature[:, tidx, :, :]
    times = times[tidx]

    return feature, times


def compute_phase_feature(
    epochs,
    freq_str="2:30:1",
    representation="angle",
    n_jobs=1,
    chunk_size=5,
    time_range=None,
):
    """
    Compute phase feature from MNE Epochs.

    Important:
        Do NOT crop epochs before Morlet TFR.
        Low-frequency wavelets may be longer than a short cropped signal.
        Crop time AFTER TFR computation.

    Standard output shape:
        (freq, time, trial, channel)

    Representations
    ---------------
    angle:
        Phase angle in radians, range [-pi, pi].
        Output shape:
            (freq, time, trial, channel)

    sin_cos:
        Circular-safe real-valued representation.
        sin and cos are concatenated along the frequency axis.
        Output shape:
            (2 * freq, time, trial, channel)

    complex_unit:
        Complex unit representation exp(1j * phase).
        Output dtype is complex64.
        Output shape:
            (freq, time, trial, channel)
    """

    if representation not in VALID_PHASE_REPRESENTATIONS:
        raise ValueError(
            f"Invalid phase representation: {representation}. "
            f"Options: {sorted(VALID_PHASE_REPRESENTATIONS)}"
        )

    freqs = parse_freqs(freq_str)
    n_cycles = freqs / 2.0

    phase_chunks = []
    times = None

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
        phase_chunk = np.angle(tfr.data)
        phase_chunks.append(phase_chunk)

        if times is None:
            times = tfr.times

    phase = np.concatenate(phase_chunks, axis=2)

    # phase:
    # (trial, channel, freq, time)
    # -> standard:
    # (freq, time, trial, channel)
    phase = np.transpose(phase, (2, 3, 0, 1))

    if representation == "angle":
        feature = phase.astype(np.float32)
        out_freqs = freqs.astype(float)
        representation_note = "Phase angle in radians, range [-pi, pi]."

    elif representation == "sin_cos":
        phase_sin = np.sin(phase)
        phase_cos = np.cos(phase)

        # Concatenate along frequency axis:
        # first half = sin(phase), second half = cos(phase)
        feature = np.concatenate([phase_sin, phase_cos], axis=0).astype(np.float32)

        out_freqs = np.concatenate([freqs, freqs]).astype(float)

        representation_note = (
            "Circular-safe sin/cos phase representation. "
            "First half of frequency axis is sin(phase); second half is cos(phase)."
        )

    elif representation == "complex_unit":
        feature = np.exp(1j * phase).astype(np.complex64)
        out_freqs = freqs.astype(float)
        representation_note = "Complex unit representation exp(1j * phase)."

    else:
        raise RuntimeError(f"Unhandled representation: {representation}")

    # Crop AFTER TFR / phase representation
    feature, times = crop_feature_time(
        feature=feature,
        times=times,
        time_range=time_range,
    )

    return feature, out_freqs, times, representation_note


def compute_and_save_phase_feature(
    epochs,
    out_dir,
    subject,
    task,
    representation="angle",
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
    Compute or load phase feature package.

    Saves
    -----
    feature:
        .npy file

    metadata:
        .meta.json file

    Standard feature shape:
        (freq, time, trial, channel)
    """

    if representation not in VALID_PHASE_REPRESENTATIONS:
        raise ValueError(
            f"Invalid phase representation: {representation}. "
            f"Options: {sorted(VALID_PHASE_REPRESENTATIONS)}"
        )

    feature_path, meta_path = build_feature_paths(
        out_dir=out_dir,
        subject=subject,
        task=task,
        feature_space="phase",
        representation=representation,
        suffix=suffix,
    )

    if not should_compute(feature_path, overwrite=overwrite):
        print(f"[SKIP] Phase feature exists: {feature_path}")
        return load_feature_package(
            feature_path=feature_path,
            meta_path=meta_path,
            mmap=True,
        )

    print("[PHASE] Computing phase feature")
    print("Subject:", subject)
    print("Task:", task)
    print("Representation:", representation)
    print("Frequency string:", freq_str)
    print("Time range:", time_range)

    feature, freqs, times, representation_note = compute_phase_feature(
        epochs=epochs,
        freq_str=freq_str,
        representation=representation,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        time_range=time_range,
    )

    notes = (
        f"{representation_note} "
        f"Computed using Morlet wavelets with n_cycles=freq/2. "
        f"TFR was computed before time cropping to avoid short-signal wavelet errors. "
        f"Pure phase feature only; circular geometry should be handled downstream."
    )

    meta = build_feature_meta(
        subject=subject,
        task=task,
        feature_space="phase",
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
    meta["tfr_crop_strategy"] = "compute_tfr_first_then_crop_feature_time"

    if representation == "sin_cos":
        n_base_freq = len(parse_freqs(freq_str))
        meta["phase_axis_info"] = {
            "type": "sin_cos_concatenated_on_frequency_axis",
            "n_base_freq": n_base_freq,
            "sin_indices": [0, n_base_freq],
            "cos_indices": [n_base_freq, 2 * n_base_freq],
            "note": "freq axis first half = sin(phase), second half = cos(phase)",
        }

    save_feature_package(
        feature_path=feature_path,
        meta_path=meta_path,
        feature=feature,
        meta=meta,
    )

    print(f"[PHASE] Saved feature: {feature_path}")
    print(f"[PHASE] Saved metadata: {meta_path}")

    return feature, meta


def load_phase_feature(
    feature_path,
    meta_path=None,
    mmap=True,
):
    """
    Load phase feature package.

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

    if meta["feature_space"] != "phase":
        raise ValueError(
            f"Expected feature_space='phase', got {meta['feature_space']}"
        )

    return feature, meta
