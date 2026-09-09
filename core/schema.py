# svm/core/schema.py

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np


FEATURE_SHAPE_FORMAT = "(freq, time, trial, channel)"

VALID_FEATURE_SPACES = {
    "raw",
    "power",
    "phase",
}

VALID_REPRESENTATIONS = {
    "raw": {
        "amplitude",
    },

    "power": {
        "magnitude",
        "magnitude_squared",
        "log_power",
    },

    "phase": {
        "angle",
        "sin_cos",
        "unit_sin_cos",
        "complex_unit",
        "resultant",
    },
}

@dataclass
class FeatureMeta:
    """
    Standard metadata for all SVM feature files.

    Required feature array shape:
        feature.shape = (freq, time, trial, channel)

    This schema is task-general and should work for:
        - recognition memory
        - picture naming
        - visual tasks
        - any future SVM classifier
    """

    subject: str
    task: str
    feature_space: str
    representation: str

    shape: List[int]
    shape_format: str

    freqs: List[float]
    times: List[float]
    ch_names: List[str]

    trial_ids: Optional[List[Any]] = None
    labels: Optional[Dict[str, List[Any]]] = None

    source_epoch_file: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_feature_meta(
    subject: str,
    task: str,
    feature_space: str,
    representation: str,
    feature: np.ndarray,
    freqs: List[float],
    times: List[float],
    ch_names: List[str],
    trial_ids: Optional[List[Any]] = None,
    labels: Optional[Dict[str, List[Any]]] = None,
    source_epoch_file: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build standardized metadata dictionary for a feature array.
    """

    validate_feature_array(feature)
    validate_feature_type(feature_space, representation)

    meta = FeatureMeta(
        subject=subject,
        task=task,
        feature_space=feature_space,
        representation=representation,
        shape=list(feature.shape),
        shape_format=FEATURE_SHAPE_FORMAT,
        freqs=list(map(float, freqs)),
        times=list(map(float, times)),
        ch_names=list(ch_names),
        trial_ids=trial_ids,
        labels=labels,
        source_epoch_file=source_epoch_file,
        notes=notes,
    )

    validate_feature_meta(meta.to_dict(), feature)

    return meta.to_dict()


def validate_feature_array(feature: np.ndarray) -> None:
    """
    Check that feature uses the standard shape:
        (freq, time, trial, channel)
    """

    if not isinstance(feature, np.ndarray):
        raise TypeError("feature must be a numpy array")

    if feature.ndim != 4:
        raise ValueError(
            f"feature must be 4D with shape {FEATURE_SHAPE_FORMAT}, "
            f"but got shape {feature.shape}"
        )

    n_freq, n_time, n_trial, n_channel = feature.shape

    if n_freq <= 0:
        raise ValueError("feature has zero frequencies")

    if n_time <= 0:
        raise ValueError("feature has zero time points")

    if n_trial <= 0:
        raise ValueError("feature has zero trials")

    if n_channel <= 0:
        raise ValueError("feature has zero channels")


def validate_feature_type(feature_space: str, representation: str) -> None:
    """
    Check whether feature_space and representation are allowed.
    """

    if feature_space not in VALID_FEATURE_SPACES:
        raise ValueError(
            f"Invalid feature_space: {feature_space}. "
            f"Options: {sorted(VALID_FEATURE_SPACES)}"
        )

    valid_reps = VALID_REPRESENTATIONS[feature_space]

    if representation not in valid_reps:
        raise ValueError(
            f"Invalid representation '{representation}' for feature_space '{feature_space}'. "
            f"Options: {sorted(valid_reps)}"
        )


def validate_feature_meta(meta: Dict[str, Any], feature: Optional[np.ndarray] = None) -> None:
    """
    Validate metadata consistency.
    """

    required_keys = [
        "subject",
        "task",
        "feature_space",
        "representation",
        "shape",
        "shape_format",
        "freqs",
        "times",
        "ch_names",
    ]

    for key in required_keys:
        if key not in meta:
            raise KeyError(f"Missing required metadata key: {key}")

    if meta["shape_format"] != FEATURE_SHAPE_FORMAT:
        raise ValueError(
            f"Invalid shape_format: {meta['shape_format']}. "
            f"Expected {FEATURE_SHAPE_FORMAT}"
        )

    validate_feature_type(meta["feature_space"], meta["representation"])

    shape = meta["shape"]

    if len(shape) != 4:
        raise ValueError(f"meta['shape'] must have length 4, got {shape}")

    n_freq, n_time, n_trial, n_channel = shape

    if len(meta["freqs"]) != n_freq:
        raise ValueError(
            f"freq length mismatch: len(freqs)={len(meta['freqs'])}, "
            f"but feature has n_freq={n_freq}"
        )

    if len(meta["times"]) != n_time:
        raise ValueError(
            f"time length mismatch: len(times)={len(meta['times'])}, "
            f"but feature has n_time={n_time}"
        )

    if len(meta["ch_names"]) != n_channel:
        raise ValueError(
            f"channel length mismatch: len(ch_names)={len(meta['ch_names'])}, "
            f"but feature has n_channel={n_channel}"
        )

    if meta.get("trial_ids") is not None:
        if len(meta["trial_ids"]) != n_trial:
            raise ValueError(
                f"trial_ids length mismatch: len(trial_ids)={len(meta['trial_ids'])}, "
                f"but feature has n_trial={n_trial}"
            )

    if meta.get("labels") is not None:
        for label_name, values in meta["labels"].items():
            if len(values) != n_trial:
                raise ValueError(
                    f"label '{label_name}' length mismatch: len={len(values)}, "
                    f"but feature has n_trial={n_trial}"
                )

    if feature is not None:
        if list(feature.shape) != list(shape):
            raise ValueError(
                f"feature shape mismatch: feature.shape={feature.shape}, "
                f"meta shape={shape}"
            )


def describe_feature(meta: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of a feature file.
    """

    shape = meta["shape"]

    return (
        f"Feature summary\n"
        f"---------------\n"
        f"Subject: {meta['subject']}\n"
        f"Task: {meta['task']}\n"
        f"Feature space: {meta['feature_space']}\n"
        f"Representation: {meta['representation']}\n"
        f"Shape format: {meta['shape_format']}\n"
        f"Shape: {shape}\n"
        f"Frequencies: {len(meta['freqs'])}\n"
        f"Time points: {len(meta['times'])}\n"
        f"Trials: {shape[2]}\n"
        f"Channels: {shape[3]}\n"
        f"Source file: {meta.get('source_epoch_file')}\n"
    )
