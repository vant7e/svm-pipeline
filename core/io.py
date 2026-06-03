# svm/core/io.py

import json
import numpy as np
from pathlib import Path

from .schema import (
    validate_feature_array,
    validate_feature_meta,
)


# =================================================
# Save feature array
# =================================================
def save_feature_array(path, feature):
    """
    Save feature array (.npy)

    Required shape:
        (freq, time, trial, channel)
    """

    validate_feature_array(feature)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if np.iscomplexobj(feature):
        np.save(path, feature.astype(np.complex64))
    else:
        np.save(path, feature.astype(np.float32))


# =================================================
# Load feature array
# =================================================
def load_feature_array(path, mmap=True):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if mmap:
        feature = np.load(path, mmap_mode="r")
    else:
        feature = np.load(path)

    validate_feature_array(feature)

    return feature


# =================================================
# Save metadata
# =================================================
def save_feature_meta(path, meta):
    """
    Save metadata as JSON

    Example:
        feature.npy
        feature.meta.json
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    validate_feature_meta(meta)

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# =================================================
# Load metadata
# =================================================
def load_feature_meta(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        meta = json.load(f)

    validate_feature_meta(meta)

    return meta


# =================================================
# Save feature package
# =================================================
def save_feature_package(
    feature_path,
    meta_path,
    feature,
    meta
):
    """
    Save:
        feature.npy
        feature.meta.json
    """

    validate_feature_array(feature)
    validate_feature_meta(meta, feature)

    save_feature_array(feature_path, feature)
    save_feature_meta(meta_path, meta)


# =================================================
# Load feature package
# =================================================
def load_feature_package(
    feature_path,
    meta_path,
    mmap=True
):
    """
    Load:
        feature.npy
        feature.meta.json
    """

    feature = load_feature_array(feature_path, mmap=mmap)
    meta = load_feature_meta(meta_path)

    validate_feature_meta(meta, feature)

    return feature, meta


# =================================================
# Check whether recomputation needed
# =================================================
def should_compute(path, overwrite=False):

    path = Path(path)

    if overwrite:
        return True

    return not path.exists()


# =================================================
# Standardized feature naming
# =================================================
def build_feature_filename(
    subject,
    task,
    feature_space,
    representation,
    suffix=None
):
    """
    Example:
        10665_recognition_raw_amplitude.npy

        10665_recognition_phase_angle.npy

        10665_recognition_power_log_power.npy
    """

    name = (
        f"{subject}"
        f"_{task}"
        f"_{feature_space}"
        f"_{representation}"
    )

    if suffix is not None:
        name += f"_{suffix}"

    return name


# =================================================
# Build feature paths
# =================================================
def build_feature_paths(
    out_dir,
    subject,
    task,
    feature_space,
    representation,
    suffix=None
):

    out_dir = Path(out_dir)

    base = build_feature_filename(
        subject=subject,
        task=task,
        feature_space=feature_space,
        representation=representation,
        suffix=suffix
    )

    feature_path = out_dir / f"{base}.npy"
    meta_path = out_dir / f"{base}.meta.json"

    return feature_path, meta_path


# =================================================
# Human-readable summary
# =================================================
def print_feature_summary(meta):

    shape = meta["shape"]

    print("\n==============================")
    print("Feature Summary")
    print("==============================")

    print("Subject:", meta["subject"])
    print("Task:", meta["task"])

    print("Feature space:", meta["feature_space"])
    print("Representation:", meta["representation"])

    print("Shape format:", meta["shape_format"])
    print("Shape:", shape)

    print("Freqs:", len(meta["freqs"]))
    print("Times:", len(meta["times"]))

    print("Trials:", shape[2])
    print("Channels:", shape[3])

    print("==============================\n")