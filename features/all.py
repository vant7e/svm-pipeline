#!/usr/bin/env python3
"""
Unified extraction of EEG raw, wavelet power, and phase features.

For each subject, this script processes the paper-matched datasets in the
fixed order defined by DATASET_ORDER, typically:

    all_id
    a_id
    c_id

Core design
-----------
Each subject x dataset is loaded ONCE.

From the loaded epochs:

1. Raw ERP amplitude is extracted directly from the epochs.
2. Complex Morlet coefficients are computed ONCE per frequency chunk.
3. Power and phase representations are derived from the same complex chunk.
4. Single-trial and identity-within-cv_block block-average packages are saved.
5. Complex Morlet coefficients are NEVER permanently saved.
6. Features are NEVER z-scored during extraction.
7. PCA is NOT applied here. PCA must be fitted within each classifier
   training fold to avoid data leakage.

Representations
---------------

Raw
~~~
amplitude
    Raw EEG/ERP amplitude.

Power
~~~~~
power
    abs(W) ** 2

log_power
    log1p(abs(W) ** 2)

Phase
~~~~~
angle
    Single trial:
        angle(W)

    Block average:
        angle(mean(exp(i * theta)))

sin_cos
    Single trial:
        [sin(theta), cos(theta)]

    Block average:
        [mean(sin(theta)), mean(cos(theta))]

resultant
    BLOCK AVERAGE ONLY:

        abs(mean(exp(i * theta)))

    This is the resultant length / phase concentration.

IMPORTANT:
A single-trial resultant is not saved because

    abs(exp(i * theta)) == 1

for every observation and therefore contains no useful variation.

Saved schemas
-------------

Raw:
    (pseudo_frequency=1, time, observation, channel)

Power / Angle / Resultant:
    (frequency, time, observation, channel)

SinCos:
    (component_frequency, time, observation, channel)

where component_frequency is:

    sin(2 Hz), ..., sin(30 Hz),
    cos(2 Hz), ..., cos(30 Hz)

Output layout
-------------

<output-root>/<subject>/<dataset>/

    raw/
        amplitude/
            single/
            block_average/

    power/
        power/
            single/
            block_average/

        log_power/
            single/
            block_average/

    phase/
        angle/
            single/
            block_average/

        sin_cos/
            single/
            block_average/

        resultant/
            block_average/

Each package contains:

    feature.npy
    feature.meta.json
    observations.csv

The dataset also contains:

    feature_manifest.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import traceback

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from numpy.lib.format import open_memmap

from svm.features.common import (
    ANALYSIS_N_TIMEPOINTS,
    DATASET_ORDER,
    OT_CHANNELS,
    average_trial_first_array,
    build_identity_block_groups,
    dataframe_to_labels,
    load_and_combine_sessions,
    prepare_single_metadata,
    print_abnormal_repetition_counts,
    trial_channel_freq_time_to_standard,
    validate_all_a_c_consistency,
)

from svm.features.wavelet import (
    WaveletConfig,
    build_wavelet_audit_metadata,
    iter_complex_wavelet_chunks,
)


# =============================================================================
# REPRESENTATION REGISTRY
# =============================================================================

RAW_REPRESENTATIONS = (
    "amplitude",
)

POWER_REPRESENTATIONS = (
    "power",
    "log_power",
)

PHASE_REPRESENTATIONS = (
    "angle",
    "sin_cos",
    "resultant",
)

OBSERVATION_MODES = (
    "single",
    "block_average",
)


# Which observation modes are scientifically meaningful for each
# representation.
PACKAGE_MODES = {
    ("raw", "amplitude"): (
        "single",
        "block_average",
    ),

    ("power", "power"): (
        "single",
        "block_average",
    ),

    ("power", "log_power"): (
        "single",
        "block_average",
    ),

    ("phase", "angle"): (
        "single",
        "block_average",
    ),

    ("phase", "sin_cos"): (
        "single",
        "block_average",
    ),

    # Resultant is meaningful only after averaging phase vectors.
    ("phase", "resultant"): (
        "block_average",
    ),
}


FEATURE_SCHEMA = (
    "frequency, time, observation, channel"
)

RAW_SCHEMA = (
    "pseudo_frequency, time, observation, channel"
)

SIN_COS_SCHEMA = (
    "component_frequency, time, observation, channel"
)


# =============================================================================
# DATA CONTAINERS
# =============================================================================


@dataclass(frozen=True)
class PackageKey:
    """Unique identifier for one saved feature package."""

    feature_space: str
    representation: str
    observation_mode: str

    def __post_init__(self) -> None:

        registry = {
            "raw": RAW_REPRESENTATIONS,
            "power": POWER_REPRESENTATIONS,
            "phase": PHASE_REPRESENTATIONS,
        }

        if self.feature_space not in registry:
            raise ValueError(
                f"Invalid feature_space "
                f"{self.feature_space!r}."
            )

        if self.representation not in registry[
            self.feature_space
        ]:
            raise ValueError(
                f"Invalid representation "
                f"{self.representation!r} "
                f"for feature_space "
                f"{self.feature_space!r}."
            )

        valid_modes = PACKAGE_MODES.get(
            (
                self.feature_space,
                self.representation,
            )
        )

        if valid_modes is None:
            raise ValueError(
                "No PACKAGE_MODES entry for "
                f"{self.feature_space}/"
                f"{self.representation}."
            )

        if self.observation_mode not in valid_modes:
            raise ValueError(
                f"{self.feature_space}/"
                f"{self.representation} does not support "
                f"observation_mode="
                f"{self.observation_mode!r}."
            )

    @property
    def name(self) -> str:

        return (
            f"{self.feature_space}__"
            f"{self.representation}__"
            f"{self.observation_mode}"
        )


@dataclass
class PackageWriter:
    """Open on-disk NumPy writer and package metadata."""

    key: PackageKey

    package_dir: Path

    feature_path: Path
    metadata_path: Path
    observations_path: Path

    array: np.memmap

    expected_shape: tuple[
        int,
        int,
        int,
        int,
    ]

    dtype_name: str

    frequency_labels: list[str]
    frequencies_hz: list[float]

    component_labels: list[str] | None = None

    frequency_axis_is_pseudo: bool = False

    def flush(self) -> None:
        self.array.flush()


# =============================================================================
# JSON / FILE HELPERS
# =============================================================================


def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(
            value
        ):
            return None

        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item
            in value
        ]

    return value


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_safe(
                payload
            ),
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    os.replace(
        temporary_path,
        path,
    )


def remove_path(
    path: Path,
) -> None:

    path = Path(
        path
    )

    if (
        not path.exists()
        and not path.is_symlink()
    ):
        return

    if (
        path.is_dir()
        and not path.is_symlink()
    ):
        shutil.rmtree(
            path
        )

    else:
        path.unlink()


# =============================================================================
# BLOCK METADATA / GROUPING
# =============================================================================


def build_block_metadata_and_groups(
    single_metadata: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[
        tuple[
            tuple[Any, ...],
            np.ndarray,
        ]
    ],
]:

    groups = (
        build_identity_block_groups(
            single_metadata
        )
    )

    dummy = np.zeros(
        (
            len(
                single_metadata
            ),
            1,
        ),
        dtype=np.float32,
    )

    _, block_metadata = (
        average_trial_first_array(
            dummy,
            single_metadata,
            output_dtype=np.float32,
            mean_dtype=np.float64,
        )
    )

    if (
        len(
            block_metadata
        )
        != len(
            groups
        )
    ):
        raise RuntimeError(
            "Block metadata count does not "
            "match group count."
        )

    expected_keys = [
        (
            str(
                group_key[0]
            ),
            int(
                group_key[1]
            ),
            int(
                group_key[2]
            ),
        )
        for group_key, _
        in groups
    ]

    observed_keys = [
        (
            str(
                row.subject
            ),
            int(
                row.cv_block
            ),
            int(
                row.identity
            ),
        )
        for row
        in block_metadata.itertuples(
            index=False
        )
    ]

    if expected_keys != observed_keys:
        raise RuntimeError(
            "Block metadata order does not "
            "match group order."
        )

    return (
        block_metadata,
        groups,
    )


def grouped_mean(
    trial_first_data: np.ndarray,
    groups: Sequence[
        tuple[
            tuple[Any, ...],
            np.ndarray,
        ]
    ],
    *,
    output_dtype: np.dtype | type,
) -> np.ndarray:

    if trial_first_data.ndim < 1:
        raise ValueError(
            "trial_first_data must have "
            "at least one dimension."
        )

    if np.iscomplexobj(
        trial_first_data
    ):
        mean_dtype = np.complex128

    else:
        mean_dtype = np.float64

    means = []

    for _, positions in groups:

        means.append(
            trial_first_data[
                positions
            ].mean(
                axis=0,
                dtype=mean_dtype,
            )
        )

    averaged = np.stack(
        means,
        axis=0,
    ).astype(
        output_dtype,
        copy=False,
    )

    if not np.isfinite(
        averaged
    ).all():

        raise RuntimeError(
            "Grouped mean contains "
            "NaN or infinity."
        )

    return averaged


# =============================================================================
# RAW EXTRACTION
# =============================================================================


def find_matching_time_indices(
    epoch_times: np.ndarray,
    target_times: np.ndarray,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Find Epochs time indices corresponding to the exact analysis time vector.

    This lets raw ERP use exactly the same time samples as the wavelet
    features.
    """

    epoch_times = np.asarray(
        epoch_times,
        dtype=np.float64,
    )

    target_times = np.asarray(
        target_times,
        dtype=np.float64,
    )

    indices = []

    for target in target_times:

        idx = int(
            np.argmin(
                np.abs(
                    epoch_times
                    - target
                )
            )
        )

        if not np.isclose(
            epoch_times[idx],
            target,
            atol=atol,
            rtol=0,
        ):
            raise RuntimeError(
                "Could not exactly align raw epoch time "
                f"{target:.12f} s. Closest sample is "
                f"{epoch_times[idx]:.12f} s."
            )

        indices.append(
            idx
        )

    indices = np.asarray(
        indices,
        dtype=int,
    )

    if len(
        np.unique(
            indices
        )
    ) != len(
        indices
    ):
        raise RuntimeError(
            "Raw time alignment produced "
            "duplicate epoch indices."
        )

    return indices


def extract_raw_amplitude(
    epochs,
    *,
    picks: Sequence[str],
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Extract raw ERP amplitude.

    Output:
        trial x channel x time
    """

    full_data = epochs.get_data(
        picks=list(
            picks
        ),
        copy=True,
    )

    full_data = full_data.astype(
        np.float32,
        copy=False,
    )

    time_indices = (
        find_matching_time_indices(
            epochs.times,
            target_times,
        )
    )

    data = full_data[
        :,
        :,
        time_indices,
    ]

    del full_data

    if data.shape[2] != len(
        target_times
    ):
        raise RuntimeError(
            "Raw analysis time length mismatch."
        )

    if not np.isfinite(
        data
    ).all():
        raise RuntimeError(
            "Raw ERP contains NaN or infinity."
        )

    return data


def raw_trial_first_to_standard(
    data: np.ndarray,
) -> np.ndarray:
    """
    Input:
        observation x channel x time

    Output:
        pseudo_frequency=1 x time x observation x channel
    """

    if data.ndim != 3:
        raise ValueError(
            "Raw data must be "
            "observation x channel x time, "
            f"got {data.shape}."
        )

    standard = np.transpose(
        data,
        (
            2,
            0,
            1,
        ),
    )

    standard = standard[
        np.newaxis,
        ...,
    ]

    return standard


# =============================================================================
# POWER DERIVATION
# =============================================================================


def derive_power_representations(
    complex_chunk: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:

    if complex_chunk.ndim != 4:
        raise ValueError(
            "complex_chunk must have shape "
            "(trial, channel, frequency, time), "
            f"got {complex_chunk.shape}."
        )

    if not np.iscomplexobj(
        complex_chunk
    ):
        raise TypeError(
            "Power derivation requires "
            "complex Morlet coefficients."
        )

    magnitude = np.abs(
        complex_chunk
    ).astype(
        np.float32,
        copy=False,
    )

    power = np.square(
        magnitude,
        dtype=np.float32,
    )

    del magnitude

    log_power = np.log1p(
        power
    ).astype(
        np.float32,
        copy=False,
    )

    outputs = {
        "power": power,
        "log_power": log_power,
    }

    for name, array in outputs.items():

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"Power representation "
                f"{name!r} contains "
                "NaN or infinity."
            )

    return outputs


# =============================================================================
# PHASE DERIVATION
# =============================================================================


def derive_phase_single(
    complex_chunk: np.ndarray,
) -> tuple[
    dict[
        str,
        np.ndarray,
    ],
    np.ndarray,
]:
    """
    Derive single-trial Angle / SinCos and an internal unit-complex array.

    Returns
    -------
    phase_single
        angle and sin_cos only.

    unit_complex
        exp(i * theta), used internally for circular block averaging.
    """

    if complex_chunk.ndim != 4:
        raise ValueError(
            "complex_chunk must have shape "
            "(trial, channel, frequency, time), "
            f"got {complex_chunk.shape}."
        )

    if not np.iscomplexobj(
        complex_chunk
    ):
        raise TypeError(
            "Phase derivation requires "
            "complex Morlet coefficients."
        )

    angle = np.angle(
        complex_chunk
    ).astype(
        np.float32,
        copy=False,
    )

    sin_phase = np.sin(
        angle
    ).astype(
        np.float32,
        copy=False,
    )

    cos_phase = np.cos(
        angle
    ).astype(
        np.float32,
        copy=False,
    )

    sin_cos = np.stack(
        [
            sin_phase,
            cos_phase,
        ],
        axis=2,
    ).astype(
        np.float32,
        copy=False,
    )

    unit_complex = (
        cos_phase.astype(
            np.complex64,
            copy=False,
        )
        +
        1j
        * sin_phase.astype(
            np.complex64,
            copy=False,
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    outputs = {
        "angle": angle,
        "sin_cos": sin_cos,
    }

    for name, array in outputs.items():

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"Phase representation "
                f"{name!r} contains "
                "NaN or infinity."
            )

    if not np.isfinite(
        unit_complex
    ).all():

        raise RuntimeError(
            "Internal unit-complex phase "
            "contains NaN or infinity."
        )

    return (
        outputs,
        unit_complex,
    )


def derive_phase_block(
    unit_complex: np.ndarray,
    groups: Sequence[
        tuple[
            tuple[Any, ...],
            np.ndarray,
        ]
    ],
) -> dict[
    str,
    np.ndarray,
]:
    """
    Circular-average trial phase, then derive:

        Angle
        SinCos
        Resultant

    Input:
        trial x channel x frequency x time

    Outputs use trial-first / block-first organization.
    """

    mean_vector = grouped_mean(
        unit_complex,
        groups,
        output_dtype=np.complex64,
    )

    block_angle = np.angle(
        mean_vector
    ).astype(
        np.float32,
        copy=False,
    )

    block_sin = (
        mean_vector.imag.astype(
            np.float32,
            copy=False,
        )
    )

    block_cos = (
        mean_vector.real.astype(
            np.float32,
            copy=False,
        )
    )

    block_sin_cos = np.stack(
        [
            block_sin,
            block_cos,
        ],
        axis=2,
    ).astype(
        np.float32,
        copy=False,
    )

    block_resultant = np.abs(
        mean_vector
    ).astype(
        np.float32,
        copy=False,
    )

    # Numerical tolerance only.
    if (
        block_resultant.min()
        < -1e-6
    ):
        raise RuntimeError(
            "Resultant contains negative values."
        )

    if (
        block_resultant.max()
        > 1.0001
    ):
        raise RuntimeError(
            "Resultant exceeds expected upper bound 1."
        )

    outputs = {
        "angle": block_angle,
        "sin_cos": block_sin_cos,
        "resultant": block_resultant,
    }

    for name, array in outputs.items():

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"Block phase representation "
                f"{name!r} contains "
                "NaN or infinity."
            )

    return outputs


# =============================================================================
# PACKAGE SHAPES / METADATA AXES
# =============================================================================


def representation_shape(
    *,
    feature_space: str,
    representation: str,
    n_frequencies: int,
    n_times: int,
    n_observations: int,
    n_channels: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    if feature_space == "raw":

        frequency_axis = 1

    elif representation == "sin_cos":

        frequency_axis = (
            2
            * n_frequencies
        )

    else:
        frequency_axis = (
            n_frequencies
        )

    return (
        frequency_axis,
        n_times,
        n_observations,
        n_channels,
    )


def frequency_axis_metadata(
    *,
    feature_space: str,
    representation: str,
    frequencies: np.ndarray,
) -> tuple[
    list[str],
    list[float],
    list[str] | None,
    bool,
]:

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    if feature_space == "raw":

        return (
            [
                "raw_amplitude"
            ],
            [
                0.0
            ],
            None,
            True,
        )

    if representation == "sin_cos":

        labels = (
            [
                f"sin_{freq:g}Hz"
                for freq
                in frequencies
            ]
            +
            [
                f"cos_{freq:g}Hz"
                for freq
                in frequencies
            ]
        )

        freqs = (
            frequencies.tolist()
            +
            frequencies.tolist()
        )

        components = (
            [
                "sin"
            ]
            * len(
                frequencies
            )
            +
            [
                "cos"
            ]
            * len(
                frequencies
            )
        )

        return (
            labels,
            freqs,
            components,
            False,
        )

    return (
        [
            f"{freq:g}Hz"
            for freq
            in frequencies
        ],
        frequencies.tolist(),
        None,
        False,
    )


# =============================================================================
# PACKAGE CREATION
# =============================================================================


def prepare_package_writer(
    *,
    dataset_root: Path,
    key: PackageKey,
    frequencies: np.ndarray,
    n_times: int,
    n_observations: int,
    n_channels: int,
    overwrite: bool,
) -> PackageWriter:

    package_dir = (
        dataset_root
        / key.feature_space
        / key.representation
        / key.observation_mode
    )

    if package_dir.exists():

        if overwrite:

            remove_path(
                package_dir
            )

        else:
            raise FileExistsError(
                f"Output package already exists: "
                f"{package_dir}"
            )

    package_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    feature_path = (
        package_dir
        / "feature.npy"
    )

    metadata_path = (
        package_dir
        / "feature.meta.json"
    )

    observations_path = (
        package_dir
        / "observations.csv"
    )

    expected_shape = (
        representation_shape(
            feature_space=key.feature_space,
            representation=key.representation,
            n_frequencies=len(
                frequencies
            ),
            n_times=n_times,
            n_observations=n_observations,
            n_channels=n_channels,
        )
    )

    dtype = np.dtype(
        np.float32
    )

    array = open_memmap(
        feature_path,
        mode="w+",
        dtype=dtype,
        shape=expected_shape,
    )

    (
        frequency_labels,
        frequencies_hz,
        component_labels,
        frequency_axis_is_pseudo,
    ) = frequency_axis_metadata(
        feature_space=key.feature_space,
        representation=key.representation,
        frequencies=frequencies,
    )

    return PackageWriter(
        key=key,
        package_dir=package_dir,
        feature_path=feature_path,
        metadata_path=metadata_path,
        observations_path=observations_path,
        array=array,
        expected_shape=expected_shape,
        dtype_name=str(
            dtype
        ),
        frequency_labels=frequency_labels,
        frequencies_hz=frequencies_hz,
        component_labels=component_labels,
        frequency_axis_is_pseudo=(
            frequency_axis_is_pseudo
        ),
    )


def prepare_all_package_writers(
    *,
    dataset_root: Path,
    frequencies: np.ndarray,
    n_times: int,
    n_single_observations: int,
    n_block_observations: int,
    n_channels: int,
) -> dict[
    PackageKey,
    PackageWriter,
]:

    writers = {}

    representation_registry = {
        "raw": (
            RAW_REPRESENTATIONS
        ),
        "power": (
            POWER_REPRESENTATIONS
        ),
        "phase": (
            PHASE_REPRESENTATIONS
        ),
    }

    for (
        feature_space,
        representations,
    ) in representation_registry.items():

        for representation in representations:

            modes = PACKAGE_MODES[
                (
                    feature_space,
                    representation,
                )
            ]

            for mode in modes:

                n_observations = (
                    n_single_observations
                    if mode == "single"
                    else n_block_observations
                )

                key = PackageKey(
                    feature_space=feature_space,
                    representation=representation,
                    observation_mode=mode,
                )

                writers[
                    key
                ] = (
                    prepare_package_writer(
                        dataset_root=dataset_root,
                        key=key,
                        frequencies=frequencies,
                        n_times=n_times,
                        n_observations=n_observations,
                        n_channels=n_channels,
                        overwrite=False,
                    )
                )

    return writers


# =============================================================================
# WRITING
# =============================================================================


def write_raw_array(
    writer: PackageWriter,
    data: np.ndarray,
) -> None:

    standard = (
        raw_trial_first_to_standard(
            data
        )
    )

    if standard.shape != writer.expected_shape:

        raise RuntimeError(
            f"Raw standard shape "
            f"{standard.shape} does not match "
            f"writer shape "
            f"{writer.expected_shape}."
        )

    writer.array[
        :,
        :,
        :,
        :,
    ] = standard

    del standard


def write_standard_chunk(
    writer: PackageWriter,
    trial_first_chunk: np.ndarray,
    frequency_indices: np.ndarray,
) -> None:

    standard = (
        trial_channel_freq_time_to_standard(
            trial_first_chunk,
            dtype=None,
        )
    )

    writer.array[
        frequency_indices,
        :,
        :,
        :,
    ] = standard

    del standard


def write_sin_cos_chunk(
    writer: PackageWriter,
    trial_first_chunk: np.ndarray,
    frequency_indices: np.ndarray,
    n_total_frequencies: int,
) -> None:

    if trial_first_chunk.ndim != 5:

        raise ValueError(
            "SinCos must have shape "
            "(observation, channel, component, "
            "frequency, time)."
        )

    if trial_first_chunk.shape[
        2
    ] != 2:

        raise ValueError(
            "SinCos component axis must "
            "have length 2."
        )

    sin_standard = (
        trial_channel_freq_time_to_standard(
            trial_first_chunk[
                :,
                :,
                0,
                :,
                :,
            ],
            dtype=None,
        )
    )

    cos_standard = (
        trial_channel_freq_time_to_standard(
            trial_first_chunk[
                :,
                :,
                1,
                :,
                :,
            ],
            dtype=None,
        )
    )

    writer.array[
        frequency_indices,
        :,
        :,
        :,
    ] = sin_standard

    writer.array[
        frequency_indices
        + n_total_frequencies,
        :,
        :,
        :,
    ] = cos_standard

    del sin_standard
    del cos_standard


def write_wavelet_representation_chunk(
    *,
    writer: PackageWriter,
    trial_first_chunk: np.ndarray,
    frequency_indices: np.ndarray,
    n_total_frequencies: int,
) -> None:

    if (
        writer.key.representation
        == "sin_cos"
    ):

        write_sin_cos_chunk(
            writer=writer,
            trial_first_chunk=trial_first_chunk,
            frequency_indices=frequency_indices,
            n_total_frequencies=n_total_frequencies,
        )

    else:

        write_standard_chunk(
            writer=writer,
            trial_first_chunk=trial_first_chunk,
            frequency_indices=frequency_indices,
        )


# =============================================================================
# PACKAGE METADATA
# =============================================================================


def build_package_metadata(
    *,
    subject: str,
    dataset_name: str,
    writer: PackageWriter,
    observation_metadata: pd.DataFrame,
    source_paths: Sequence[
        Path
    ],
    wavelet_config: WaveletConfig,
    selected_times: np.ndarray,
    channel_order: Sequence[str],
    created_at_utc: str,
) -> dict[
    str,
    Any,
]:

    wavelet_audit = None

    if writer.key.feature_space != "raw":

        wavelet_audit = (
            build_wavelet_audit_metadata(
                config=wavelet_config,
                selected_times=selected_times,
                picks=list(
                    channel_order
                ),
            )
        )

    definitions = {

        ("raw", "amplitude"): (
            "Raw EEG/ERP amplitude extracted from "
            "the same analysis time samples used "
            "for wavelet features."
        ),

        ("power", "power"): (
            "Squared magnitude of the complex "
            "Morlet coefficient: abs(W) ** 2."
        ),

        ("power", "log_power"): (
            "Natural logarithm of one plus "
            "Morlet power: log1p(abs(W) ** 2)."
        ),

        ("phase", "angle"): (
            "Single: phase angle in radians. "
            "Block average: angle of the mean "
            "unit-complex phase vector."
        ),

        ("phase", "sin_cos"): (
            "Single: sine and cosine of phase. "
            "Block average: mean sine and mean "
            "cosine, equivalent to imaginary and "
            "real components of the mean "
            "unit-complex phase vector."
        ),

        ("phase", "resultant"): (
            "Block-average resultant length: "
            "magnitude of the mean unit-complex "
            "phase vector, equivalent to "
            "sqrt(mean_sin_phase^2 + "
            "mean_cos_phase^2)."
        ),
    }

    labels = dataframe_to_labels(
        observation_metadata
    )

    if writer.key.feature_space == "raw":

        schema = RAW_SCHEMA

        axis_order = [
            "pseudo_frequency",
            "time",
            "observation",
            "channel",
        ]

    elif writer.key.representation == "sin_cos":

        schema = SIN_COS_SCHEMA

        axis_order = [
            "component_frequency",
            "time",
            "observation",
            "channel",
        ]

    else:

        schema = FEATURE_SCHEMA

        axis_order = [
            "frequency",
            "time",
            "observation",
            "channel",
        ]

    metadata = {

        "format_version": "2.0",

        "created_at_utc": (
            created_at_utc
        ),

        "subject": str(
            subject
        ),

        "dataset_name": (
            dataset_name
        ),

        # Canonical field expected by newer decoder/schema code.
        "feature_space": (
            writer.key.feature_space
        ),

        # Compatibility alias for older code.
        "feature_type": (
            writer.key.feature_space
        ),

        "representation": (
            writer.key.representation
        ),

        "observation_mode": (
            writer.key.observation_mode
        ),

        "representation_definition": (
            definitions[
                (
                    writer.key.feature_space,
                    writer.key.representation,
                )
            ]
        ),

        "feature_file": (
            writer.feature_path.name
        ),

        "observation_file": (
            writer.observations_path.name
        ),

        "shape": list(
            writer.expected_shape
        ),

        "feature_shape": list(
            writer.expected_shape
        ),

        "dtype": (
            writer.dtype_name
        ),

        "schema": (
            schema
        ),

        "shape_format": (
            schema
        ),

        "axis_order": (
            axis_order
        ),

        "frequencies_hz": (
            writer.frequencies_hz
        ),

        # Compatibility alias.
        "freqs": (
            writer.frequencies_hz
        ),

        "frequency_axis_labels": (
            writer.frequency_labels
        ),

        "component_labels": (
            writer.component_labels
        ),

        "frequency_axis_is_pseudo": (
            writer.frequency_axis_is_pseudo
        ),

        "times_seconds": (
            selected_times.tolist()
        ),

        # Compatibility alias.
        "times": (
            selected_times.tolist()
        ),

        "channel_order": list(
            channel_order
        ),

        # Compatibility alias.
        "ch_names": list(
            channel_order
        ),

        "n_observations": int(
            len(
                observation_metadata
            )
        ),

        "observation_id_column": (
            "observation_id"
        ),

        "labels": (
            labels
        ),

        "source_epoch_files": [
            str(
                Path(
                    path
                ).resolve()
            )
            for path
            in source_paths
        ],

        "morlet": (
            wavelet_audit
        ),

        "z_scored": False,

        "z_score_stage": (
            "Classifier cross-validation "
            "training fold only."
        ),

        "pca_applied": False,

        "pca_stage": (
            "Optional classifier "
            "cross-validation training fold only."
        ),

        "block_average_rule": (
            "Arithmetic mean of available valid "
            "repetitions within "
            "subject x cv_block x identity. "
            "No zero-padding."
            if (
                writer.key.observation_mode
                == "block_average"
            )
            else None
        ),

        "phase_block_average_rule": (
            "Circular averaging through the "
            "mean unit-complex phase vector."
            if (
                writer.key.feature_space
                == "phase"
                and writer.key.observation_mode
                == "block_average"
            )
            else None
        ),

        "complex_morlet_coefficients_saved": (
            False
        ),
    }

    if (
        writer.key.feature_space
        == "phase"
        and writer.key.representation
        == "resultant"
    ):

        metadata[
            "derived_from_representation"
        ] = "unit_complex_internal"

        metadata[
            "resultant_definition"
        ] = (
            "sqrt(mean_sin_phase^2 + "
            "mean_cos_phase^2)"
        )

        metadata[
            "resultant_derivation_stage"
        ] = (
            "after averaging unit-complex phase "
            "vectors within identity x cv_block"
        )

    return metadata


# =============================================================================
# FINALIZATION
# =============================================================================


def finalize_packages(
    *,
    writers: dict[
        PackageKey,
        PackageWriter,
    ],
    subject: str,
    dataset_name: str,
    single_metadata: pd.DataFrame,
    block_metadata: pd.DataFrame,
    source_paths: Sequence[
        Path
    ],
    wavelet_config: WaveletConfig,
    selected_times: np.ndarray,
    channel_order: Sequence[str],
    created_at_utc: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    manifest_packages = []

    for key, writer in writers.items():

        writer.flush()

        observation_metadata = (
            single_metadata
            if (
                key.observation_mode
                == "single"
            )
            else block_metadata
        )

        observation_metadata.to_csv(
            writer.observations_path,
            index=False,
        )

        metadata = (
            build_package_metadata(
                subject=subject,
                dataset_name=dataset_name,
                writer=writer,
                observation_metadata=observation_metadata,
                source_paths=source_paths,
                wavelet_config=wavelet_config,
                selected_times=selected_times,
                channel_order=channel_order,
                created_at_utc=created_at_utc,
            )
        )

        write_json(
            writer.metadata_path,
            metadata,
        )

        verification = np.load(
            writer.feature_path,
            mmap_mode="r",
        )

        if (
            verification.shape
            != writer.expected_shape
        ):

            raise RuntimeError(
                f"{writer.feature_path}: "
                f"expected shape "
                f"{writer.expected_shape}, "
                f"got {verification.shape}."
            )

        if (
            str(
                verification.dtype
            )
            != writer.dtype_name
        ):

            raise RuntimeError(
                f"{writer.feature_path}: "
                f"expected dtype "
                f"{writer.dtype_name}, "
                f"got "
                f"{verification.dtype}."
            )

        manifest_packages.append(
            {
                "name": (
                    key.name
                ),

                "feature_space": (
                    key.feature_space
                ),

                "representation": (
                    key.representation
                ),

                "observation_mode": (
                    key.observation_mode
                ),

                "relative_directory": str(
                    writer.package_dir.relative_to(
                        writer.package_dir.parents[
                            3
                        ]
                    )
                ),

                "shape": list(
                    writer.expected_shape
                ),

                "dtype": (
                    writer.dtype_name
                ),

                "feature_bytes": int(
                    writer.feature_path.stat().st_size
                ),
            }
        )

        del verification

    return manifest_packages


def close_writers(
    writers: dict[
        PackageKey,
        PackageWriter,
    ],
) -> None:

    for writer in writers.values():

        try:
            writer.flush()
        except Exception:
            pass

        try:
            del writer.array
        except Exception:
            pass

    gc.collect()


# =============================================================================
# DATASET PROCESSING
# =============================================================================


def process_dataset(
    *,
    subject: str,
    dataset_name: str,
    preprocessed_root: Path,
    output_root: Path,
    wavelet_config: WaveletConfig,
    overwrite: bool,
    expected_repetitions: int,
) -> dict[
    str,
    pd.DataFrame,
]:

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"SUBJECT={subject} | "
        f"DATASET={dataset_name}"
    )

    print(
        "=" * 90
    )

    dataset_root = (
        Path(
            output_root
        )
        / str(
            subject
        )
        / dataset_name
    )

    if dataset_root.exists():

        if overwrite:

            print(
                "[OVERWRITE] Removing: "
                f"{dataset_root}"
            )

            remove_path(
                dataset_root
            )

        else:

            raise FileExistsError(
                f"Dataset output already exists: "
                f"{dataset_root}. "
                "Use --overwrite."
            )

    dataset_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    incomplete_marker = (
        dataset_root
        / "_INCOMPLETE"
    )

    incomplete_marker.write_text(
        "Feature extraction did not "
        "complete successfully.\n",
        encoding="utf-8",
    )

    writers = {}

    try:

        # ============================================================
        # LOAD EPOCHS ONCE
        # ============================================================

        epochs, source_paths = (
            load_and_combine_sessions(
                preprocessed_root=preprocessed_root,
                subject=subject,
                dataset_name=dataset_name,
                preload=True,
                verbose="INFO",
            )
        )

        if epochs.metadata is None:

            raise RuntimeError(
                "Combined Epochs metadata "
                "is unexpectedly missing."
            )

        single_metadata = (
            prepare_single_metadata(
                epochs.metadata
            )
        )

        print_abnormal_repetition_counts(
            single_metadata,
            expected_count=expected_repetitions,
        )

        (
            block_metadata,
            groups,
        ) = build_block_metadata_and_groups(
            single_metadata
        )

        frequencies = (
            wavelet_config.frequencies()
        )

        n_frequencies = len(
            frequencies
        )

        n_times = (
            wavelet_config.analysis_n_timepoints
        )

        n_channels = len(
            OT_CHANNELS
        )

        # ============================================================
        # PREPARE ALL OUTPUT PACKAGES
        # ============================================================

        writers = (
            prepare_all_package_writers(
                dataset_root=dataset_root,
                frequencies=frequencies,
                n_times=n_times,
                n_single_observations=len(
                    single_metadata
                ),
                n_block_observations=len(
                    block_metadata
                ),
                n_channels=n_channels,
            )
        )

        selected_times = None

        completed_frequency_indices = []

        raw_written = False

        # ============================================================
        # MORLET CHUNKS
        # ============================================================

        for wavelet_chunk in (
            iter_complex_wavelet_chunks(
                epochs=epochs,
                config=wavelet_config,
                picks=OT_CHANNELS,
            )
        ):

            chunk_indices = (
                wavelet_chunk.frequency_indices
            )

            complex_chunk = (
                wavelet_chunk.data
            )

            # --------------------------------------------------------
            # Establish shared analysis time vector.
            # --------------------------------------------------------

            if selected_times is None:

                selected_times = (
                    wavelet_chunk.times.copy()
                )

            elif not np.array_equal(
                selected_times,
                wavelet_chunk.times,
            ):

                raise RuntimeError(
                    "Time vectors differ "
                    "across Morlet chunks."
                )

            # ========================================================
            # RAW
            #
            # Do this once, when selected_times becomes available.
            # ========================================================

            if not raw_written:

                print(
                    "[RAW] Extracting raw amplitude "
                    "using the same analysis time samples."
                )

                raw_single = (
                    extract_raw_amplitude(
                        epochs,
                        picks=OT_CHANNELS,
                        target_times=selected_times,
                    )
                )

                raw_single_key = (
                    PackageKey(
                        "raw",
                        "amplitude",
                        "single",
                    )
                )

                write_raw_array(
                    writers[
                        raw_single_key
                    ],
                    raw_single,
                )

                raw_block = grouped_mean(
                    raw_single,
                    groups,
                    output_dtype=np.float32,
                )

                raw_block_key = (
                    PackageKey(
                        "raw",
                        "amplitude",
                        "block_average",
                    )
                )

                write_raw_array(
                    writers[
                        raw_block_key
                    ],
                    raw_block,
                )

                del raw_block
                del raw_single

                raw_written = True

                print(
                    "[RAW WRITE OK] "
                    "single + block_average"
                )

                gc.collect()

            # ========================================================
            # POWER
            # ========================================================

            power_single = (
                derive_power_representations(
                    complex_chunk
                )
            )

            for (
                representation,
                single_array,
            ) in power_single.items():

                single_key = (
                    PackageKey(
                        "power",
                        representation,
                        "single",
                    )
                )

                write_wavelet_representation_chunk(
                    writer=writers[
                        single_key
                    ],
                    trial_first_chunk=single_array,
                    frequency_indices=chunk_indices,
                    n_total_frequencies=n_frequencies,
                )

                block_array = (
                    grouped_mean(
                        single_array,
                        groups,
                        output_dtype=np.float32,
                    )
                )

                block_key = (
                    PackageKey(
                        "power",
                        representation,
                        "block_average",
                    )
                )

                write_wavelet_representation_chunk(
                    writer=writers[
                        block_key
                    ],
                    trial_first_chunk=block_array,
                    frequency_indices=chunk_indices,
                    n_total_frequencies=n_frequencies,
                )

                del block_array

            del power_single

            # ========================================================
            # PHASE
            # ========================================================

            (
                phase_single,
                unit_complex,
            ) = derive_phase_single(
                complex_chunk
            )

            # Single:
            #   Angle
            #   SinCos
            #
            # No Resultant single-trial package.
            for (
                representation,
                single_array,
            ) in phase_single.items():

                single_key = (
                    PackageKey(
                        "phase",
                        representation,
                        "single",
                    )
                )

                write_wavelet_representation_chunk(
                    writer=writers[
                        single_key
                    ],
                    trial_first_chunk=single_array,
                    frequency_indices=chunk_indices,
                    n_total_frequencies=n_frequencies,
                )

            # One circular block average.
            phase_block = (
                derive_phase_block(
                    unit_complex,
                    groups,
                )
            )

            # Block:
            #   Angle
            #   SinCos
            #   Resultant
            for (
                representation,
                block_array,
            ) in phase_block.items():

                block_key = (
                    PackageKey(
                        "phase",
                        representation,
                        "block_average",
                    )
                )

                write_wavelet_representation_chunk(
                    writer=writers[
                        block_key
                    ],
                    trial_first_chunk=block_array,
                    frequency_indices=chunk_indices,
                    n_total_frequencies=n_frequencies,
                )

            completed_frequency_indices.extend(
                chunk_indices.tolist()
            )

            print(
                "[WRITE OK] Frequencies "
                f"{wavelet_chunk.freqs[0]:g}-"
                f"{wavelet_chunk.freqs[-1]:g} Hz: "
                "power/log_power + "
                "angle/sin_cos/resultant."
            )

            del phase_block
            del phase_single
            del unit_complex
            del complex_chunk
            del wavelet_chunk

            for writer in writers.values():
                writer.flush()

            gc.collect()

        # ============================================================
        # FINAL QC
        # ============================================================

        if selected_times is None:

            raise RuntimeError(
                "No Morlet chunks were produced."
            )

        if not raw_written:

            raise RuntimeError(
                "Raw feature package was never written."
            )

        if completed_frequency_indices != list(
            range(
                n_frequencies
            )
        ):

            raise RuntimeError(
                "Completed frequency indices do "
                "not equal the configured range. "
                f"Got "
                f"{completed_frequency_indices}."
            )

        created_at_utc = (
            utc_now_iso()
        )

        manifest_packages = (
            finalize_packages(
                writers=writers,
                subject=subject,
                dataset_name=dataset_name,
                single_metadata=single_metadata,
                block_metadata=block_metadata,
                source_paths=source_paths,
                wavelet_config=wavelet_config,
                selected_times=selected_times,
                channel_order=OT_CHANNELS,
                created_at_utc=created_at_utc,
            )
        )

        dataset_manifest = {

            "format_version": "2.0",

            "created_at_utc": (
                created_at_utc
            ),

            "subject": str(
                subject
            ),

            "dataset_name": (
                dataset_name
            ),

            "source_epoch_files": [
                str(
                    Path(
                        path
                    ).resolve()
                )
                for path
                in source_paths
            ],

            "n_single_observations": int(
                len(
                    single_metadata
                )
            ),

            "n_block_average_observations": int(
                len(
                    block_metadata
                )
            ),

            "frequencies_hz": (
                frequencies.tolist()
            ),

            "times_seconds": (
                selected_times.tolist()
            ),

            "channels": list(
                OT_CHANNELS
            ),

            "feature_spaces": [
                "raw",
                "power",
                "phase",
            ],

            "raw_representations": list(
                RAW_REPRESENTATIONS
            ),

            "power_representations": list(
                POWER_REPRESENTATIONS
            ),

            "phase_representations": list(
                PHASE_REPRESENTATIONS
            ),

            "wavelet_config": (
                wavelet_config.as_dict()
            ),

            "packages": (
                manifest_packages
            ),

            "complex_morlet_coefficients_saved": (
                False
            ),

            "z_scored": False,

            "pca_applied": False,

            "status": (
                "complete"
            ),
        }

        write_json(
            (
                dataset_root
                / "feature_manifest.json"
            ),
            dataset_manifest,
        )

        incomplete_marker.unlink()

        print(
            f"[DATASET COMPLETE] "
            f"{subject} {dataset_name}: "
            f"{len(manifest_packages)} "
            "packages written."
        )

        return {
            "single": (
                single_metadata
            ),
            "block_average": (
                block_metadata
            ),
        }

    except Exception:

        failure_payload = {

            "subject": str(
                subject
            ),

            "dataset_name": (
                dataset_name
            ),

            "failed_at_utc": (
                utc_now_iso()
            ),

            "traceback": (
                traceback.format_exc()
            ),
        }

        write_json(
            (
                dataset_root
                / "feature_failure.json"
            ),
            failure_payload,
        )

        raise

    finally:

        close_writers(
            writers
        )

        try:
            del epochs
        except UnboundLocalError:
            pass

        gc.collect()


# =============================================================================
# SUBJECT PROCESSING
# =============================================================================


def process_subject(
    *,
    subject: str,
    preprocessed_root: Path,
    output_root: Path,
    wavelet_config: WaveletConfig,
    overwrite: bool,
    expected_repetitions: int,
) -> None:

    outputs = {}

    for dataset_name in DATASET_ORDER:

        outputs[
            dataset_name
        ] = process_dataset(
            subject=subject,
            dataset_name=dataset_name,
            preprocessed_root=preprocessed_root,
            output_root=output_root,
            wavelet_config=wavelet_config,
            overwrite=overwrite,
            expected_repetitions=expected_repetitions,
        )

    validate_all_a_c_consistency(
        outputs,
        modes=OBSERVATION_MODES,
    )

    subject_root = (
        Path(
            output_root
        )
        / str(
            subject
        )
    )

    subject_manifest = {

        "format_version": "2.0",

        "created_at_utc": (
            utc_now_iso()
        ),

        "subject": str(
            subject
        ),

        "dataset_order": list(
            DATASET_ORDER
        ),

        "feature_spaces": [
            "raw",
            "power",
            "phase",
        ],

        "raw_representations": list(
            RAW_REPRESENTATIONS
        ),

        "power_representations": list(
            POWER_REPRESENTATIONS
        ),

        "phase_representations": list(
            PHASE_REPRESENTATIONS
        ),

        "package_modes": {
            (
                f"{feature_space}/"
                f"{representation}"
            ): list(
                modes
            )
            for (
                feature_space,
                representation,
            ), modes
            in PACKAGE_MODES.items()
        },

        "wavelet_config": (
            wavelet_config.as_dict()
        ),

        "all_id_equals_a_id_union_c_id": (
            True
        ),

        "status": (
            "complete"
        ),
    }

    write_json(
        (
            subject_root
            / "feature_subject_manifest.json"
        ),
        subject_manifest,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"SUBJECT COMPLETE: "
        f"{subject}"
    )

    print(
        "=" * 90
    )


# =============================================================================
# CLI
# =============================================================================


def parse_subjects(
    values: Sequence[str],
) -> list[str]:

    subjects = []

    for value in values:

        for item in str(
            value
        ).split(
            ","
        ):

            item = item.strip()

            if item:
                subjects.append(
                    item
                )

    if not subjects:

        raise ValueError(
            "At least one subject "
            "must be supplied."
        )

    series = pd.Series(
        subjects
    )

    duplicated = (
        series.duplicated(
            keep=False
        )
    )

    if duplicated.any():

        duplicate_subjects = (
            series[
                duplicated
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate subjects supplied: "
            f"{duplicate_subjects}"
        )

    return subjects


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Extract unified Raw / Power / "
            "LogPower / Angle / SinCos / "
            "Resultant features."
        )
    )

    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--subject",
        action="append",
        required=True,
    )

    parser.add_argument(
        "--freq-start",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--freq-stop",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--freq-step",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--n-cycles-divisor",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--expected-repetitions",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    args = parse_args()

    subjects = parse_subjects(
        args.subject
    )

    preprocessed_root = (
        args.preprocessed_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not preprocessed_root.is_dir():

        raise NotADirectoryError(
            "Preprocessed root does "
            "not exist: "
            f"{preprocessed_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    wavelet_config = WaveletConfig(

        freq_start_hz=(
            args.freq_start
        ),

        freq_stop_hz=(
            args.freq_stop
        ),

        freq_step_hz=(
            args.freq_step
        ),

        n_cycles_divisor=(
            args.n_cycles_divisor
        ),

        frequency_chunk_size=(
            args.chunk_size
        ),

        analysis_n_timepoints=(
            ANALYSIS_N_TIMEPOINTS
        ),

        n_jobs=(
            args.n_jobs
        ),
    )

    print(
        "=" * 90
    )

    print(
        "UNIFIED EEG FEATURE EXTRACTION"
    )

    print(
        f"Subjects: "
        f"{subjects}"
    )

    print(
        f"Preprocessed root: "
        f"{preprocessed_root}"
    )

    print(
        f"Output root: "
        f"{output_root}"
    )

    print(
        "Dataset order: "
        + ", ".join(
            DATASET_ORDER
        )
    )

    print(
        "Raw representations: "
        + ", ".join(
            RAW_REPRESENTATIONS
        )
    )

    print(
        "Power representations: "
        + ", ".join(
            POWER_REPRESENTATIONS
        )
    )

    print(
        "Phase representations: "
        + ", ".join(
            PHASE_REPRESENTATIONS
        )
    )

    print(
        "Morlet config: "
        + json.dumps(
            wavelet_config.as_dict(),
            ensure_ascii=False,
        )
    )

    print(
        "=" * 90
    )

    failures = []

    for subject in subjects:

        try:

            process_subject(

                subject=subject,

                preprocessed_root=(
                    preprocessed_root
                ),

                output_root=(
                    output_root
                ),

                wavelet_config=(
                    wavelet_config
                ),

                overwrite=(
                    args.overwrite
                ),

                expected_repetitions=(
                    args.expected_repetitions
                ),
            )

        except Exception as error:

            failures.append(
                {
                    "subject": (
                        subject
                    ),

                    "error_type": (
                        type(
                            error
                        ).__name__
                    ),

                    "error": (
                        str(
                            error
                        )
                    ),
                }
            )

            print(
                f"\n[SUBJECT FAILED] "
                f"{subject}: "
                f"{type(error).__name__}: "
                f"{error}",
                file=sys.stderr,
            )

            traceback.print_exc()

    run_manifest = {

        "format_version": "2.0",

        "completed_at_utc": (
            utc_now_iso()
        ),

        "subjects_requested": (
            subjects
        ),

        "subjects_failed": (
            failures
        ),

        "subjects_completed": [
            subject
            for subject
            in subjects
            if subject
            not in {
                item[
                    "subject"
                ]
                for item
                in failures
            }
        ],

        "preprocessed_root": str(
            preprocessed_root
        ),

        "output_root": str(
            output_root
        ),

        "wavelet_config": (
            wavelet_config.as_dict()
        ),

        "dataset_order": list(
            DATASET_ORDER
        ),

        "feature_spaces": [
            "raw",
            "power",
            "phase",
        ],
    }

    write_json(
        (
            output_root
            / "feature_extraction_run.json"
        ),
        run_manifest,
    )

    if failures:

        raise RuntimeError(
            f"{len(failures)} subject(s) "
            "failed. See "
            "feature_extraction_run.json."
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ALL REQUESTED SUBJECTS COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
