#!/usr/bin/env python3
"""
Shared EEG data-organization utilities for raw, power, and phase extraction.

This module contains feature-space-independent logic shared by:

    extract_raw_features.py
    extract_wavelet_features.py

Responsibilities
----------------
* Define paper-matched datasets and occipitotemporal channels.
* Build expected S1/S2 final-Epochs paths.
* Load, validate, concatenate, and stably sort S1/S2 Epochs.
* Select the exact 307 analysis samples used across raw/power/phase.
* Convert observation tables to JSON-safe labels.
* Create stable single-trial and block-average observation IDs.
* Build identity-within-cv_block averaging groups.
* Average arbitrary trial-first real or complex arrays without zero-padding.
* Construct block-average observation metadata.
* Validate that all_id equals the union of a_id and c_id.

This module does not compute features, Morlet transforms, save packages,
z-score data, or run classifiers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
import mne
import numpy as np
import pandas as pd


# =============================================================================
# PAPER-MATCHED CONSTANTS
# =============================================================================

EXPECTED_SFREQ = 512.0
ANALYSIS_WINDOW_START_SECONDS = 0.050
ANALYSIS_N_TIMEPOINTS = 307

OT_CHANNELS = [
    "P5", "P7", "P9", "PO3", "PO7", "O1",
    "P6", "P8", "P10", "PO4", "PO8", "O2",
]

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "all_id": {
        "filename_stem": "ALL_id01-60",
        "identity_min": 1,
        "identity_max": 60,
    },
    "a_id": {
        "filename_stem": "a_id01-30",
        "identity_min": 1,
        "identity_max": 30,
    },
    "c_id": {
        "filename_stem": "c_id31-60",
        "identity_min": 31,
        "identity_max": 60,
    },
}

DATASET_ORDER = ("all_id", "a_id", "c_id")
SESSION_ORDER = ("S1", "S2")

REQUIRED_METADATA_COLUMNS = {
    "subject",
    "session",
    "session_number",
    "block_code",
    "block_within_session",
    "cv_block",
    "trial_within_block",
    "identity",
    "race",
    "original_event_code",
    "identity_repetition_within_block",
    "is_oddball",
}

BLOCK_GROUP_COLUMNS = ["subject", "cv_block", "identity"]

BLOCK_CONSTANT_COLUMNS = [
    "subject",
    "session",
    "session_number",
    "block_code",
    "block_within_session",
    "cv_block",
    "identity",
    "race",
    "original_event_code",
    "is_oddball",
]

SINGLE_SORT_COLUMNS = [
    "session_number",
    "cv_block",
    "trial_within_block",
    "identity",
    "identity_repetition_within_block",
]

BLOCK_SORT_COLUMNS = ["cv_block", "identity"]


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def json_safe(value: Any) -> Any:
    """Convert NumPy/pandas scalar values into JSON-safe Python values."""

    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def dataframe_to_labels(metadata: pd.DataFrame) -> dict[str, list[Any]]:
    """Convert an observation table into schema-compatible label lists."""

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame, "
            f"got {type(metadata).__name__}."
        )

    return {
        str(column): [json_safe(value) for value in metadata[column].tolist()]
        for column in metadata.columns
    }


def require_file(path: Path) -> Path:
    """Return a resolved file path or raise FileNotFoundError."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    return path


def validate_dataset_name(dataset_name: str) -> str:
    """Validate and normalize a dataset name."""

    dataset_name = str(dataset_name)
    if dataset_name not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Expected one of {list(DATASET_SPECS)}."
        )
    return dataset_name


# =============================================================================
# OBSERVATION IDENTIFIERS
# =============================================================================


def make_trial_ids(metadata: pd.DataFrame, mode: str) -> list[str]:
    """Create stable, unique single-trial or block-average IDs."""

    if mode not in {"single", "block_average"}:
        raise ValueError(
            f"Unknown observation mode {mode!r}; "
            "expected 'single' or 'block_average'."
        )

    required = {"subject", "cv_block", "identity"}
    if mode == "single":
        required |= {
            "session",
            "trial_within_block",
            "identity_repetition_within_block",
        }

    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Cannot construct {mode} IDs; missing columns: {sorted(missing)}"
        )

    trial_ids: list[str] = []

    for _, row in metadata.iterrows():
        subject = str(row["subject"])
        cv_block = int(row["cv_block"])
        identity = int(row["identity"])

        if mode == "single":
            session = str(row["session"])
            trial_within_block = int(row["trial_within_block"])
            repetition = int(row["identity_repetition_within_block"])
            trial_id = (
                f"{subject}_{session}_cvblock-{cv_block:02d}_"
                f"identity-{identity:02d}_trial-{trial_within_block:03d}_"
                f"rep-{repetition}"
            )
        else:
            trial_id = (
                f"{subject}_cvblock-{cv_block:02d}_"
                f"identity-{identity:02d}_average"
            )

        trial_ids.append(trial_id)

    duplicate_mask = pd.Series(trial_ids).duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = pd.Series(trial_ids)[duplicate_mask].tolist()
        raise RuntimeError(
            "Observation IDs are not unique. "
            f"First duplicates: {duplicates[:10]}"
        )

    return trial_ids


def prepare_single_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add single-trial audit fields and stable observation IDs."""

    output = metadata.reset_index(drop=True).copy()
    output["trial_mode"] = "single"
    output["n_trials_averaged"] = 1

    if "observation_id" in output.columns:
        output = output.drop(columns=["observation_id"])

    output.insert(0, "observation_id", make_trial_ids(output, mode="single"))
    return output


# =============================================================================
# INPUT DISCOVERY
# =============================================================================


def build_session_path(
    preprocessed_root: Path,
    subject: str,
    session: str,
    dataset_name: str,
) -> Path:
    """Build the expected final FIF path for one dataset and session."""

    dataset_name = validate_dataset_name(dataset_name)
    session = str(session)

    if session not in SESSION_ORDER:
        raise ValueError(
            f"Unknown session {session!r}; expected one of {SESSION_ORDER}."
        )

    stem = DATASET_SPECS[dataset_name]["filename_stem"]
    return (
        Path(preprocessed_root)
        / str(subject)
        / session
        / "12_final"
        / f"{stem}_{subject}_{session}-epo.fif"
    )


# =============================================================================
# EPOCH VALIDATION
# =============================================================================


def validate_one_session_epochs(
    epochs: mne.BaseEpochs,
    subject: str,
    session: str,
    dataset_name: str,
    source_path: Path,
) -> None:
    """Validate one final S1/S2 Epochs file before concatenation."""

    dataset_name = validate_dataset_name(dataset_name)
    source_path = Path(source_path)

    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            f"{source_path}: expected MNE Epochs, "
            f"got {type(epochs).__name__}."
        )
    if len(epochs) == 0:
        raise RuntimeError(f"{source_path}: Epochs object is empty.")

    sfreq = float(epochs.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise RuntimeError(
            f"{source_path}: expected {EXPECTED_SFREQ} Hz, got {sfreq} Hz."
        )

    missing_channels = [ch for ch in OT_CHANNELS if ch not in epochs.ch_names]
    if missing_channels:
        raise RuntimeError(
            f"{source_path}: missing required OT channels: {missing_channels}"
        )

    if epochs.metadata is None:
        raise RuntimeError(f"{source_path}: Epochs metadata is missing.")

    metadata = epochs.metadata.reset_index(drop=True)
    missing_metadata = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing_metadata:
        raise RuntimeError(
            f"{source_path}: missing metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    if len(metadata) != len(epochs):
        raise RuntimeError(
            f"{source_path}: metadata rows ({len(metadata)}) do not match "
            f"epochs ({len(epochs)})."
        )

    observed_subjects = set(metadata["subject"].astype(str))
    if observed_subjects != {str(subject)}:
        raise RuntimeError(
            f"{source_path}: metadata subject values are "
            f"{sorted(observed_subjects)}; expected only {subject}."
        )

    observed_sessions = set(metadata["session"].astype(str))
    if observed_sessions != {session}:
        raise RuntimeError(
            f"{source_path}: metadata session values are "
            f"{sorted(observed_sessions)}; expected only {session}."
        )

    expected_session_number = int(session.removeprefix("S"))
    observed_session_numbers = set(metadata["session_number"].astype(int))
    if observed_session_numbers != {expected_session_number}:
        raise RuntimeError(
            f"{source_path}: session_number values "
            f"{sorted(observed_session_numbers)} do not match {session}."
        )

    expected_cv_blocks = (
        set(range(1, 17)) if session == "S1" else set(range(17, 33))
    )
    observed_cv_blocks = set(metadata["cv_block"].astype(int))
    if not observed_cv_blocks.issubset(expected_cv_blocks):
        raise RuntimeError(
            f"{source_path}: unexpected cv_block values "
            f"{sorted(observed_cv_blocks - expected_cv_blocks)}."
        )

    spec = DATASET_SPECS[dataset_name]
    identity_min = int(spec["identity_min"])
    identity_max = int(spec["identity_max"])
    identity_values = metadata["identity"].astype(int)
    valid_identity_mask = identity_values.between(identity_min, identity_max)

    if not valid_identity_mask.all():
        bad = sorted(set(identity_values[~valid_identity_mask]))
        raise RuntimeError(
            f"{source_path}: identities outside "
            f"{identity_min}-{identity_max}: {bad}"
        )

    if not np.array_equal(
        metadata["identity"].to_numpy(dtype=int),
        epochs.events[:, 2].astype(int),
    ):
        raise RuntimeError(
            f"{source_path}: metadata identity values do not match "
            "epochs.events[:, 2]."
        )

    if metadata["is_oddball"].astype(bool).any():
        raise RuntimeError(
            f"{source_path}: oddball trials unexpectedly remain."
        )

    selected_data = epochs.get_data(picks=OT_CHANNELS, copy=False)
    if not np.isfinite(selected_data).all():
        raise RuntimeError(
            f"{source_path}: selected EEG data contain NaN or infinity."
        )


def validate_combined_epochs(
    epochs: mne.BaseEpochs,
    subject: str,
    dataset_name: str,
) -> None:
    """Run post-concatenation checks shared by all feature spaces."""

    dataset_name = validate_dataset_name(dataset_name)
    if epochs.metadata is None:
        raise RuntimeError(
            f"{subject} {dataset_name}: combined metadata is missing."
        )

    metadata = epochs.metadata.reset_index(drop=True)
    if len(metadata) != len(epochs):
        raise RuntimeError(
            f"{subject} {dataset_name}: combined metadata rows "
            f"({len(metadata)}) do not match epochs ({len(epochs)})."
        )

    if set(metadata["subject"].astype(str)) != {str(subject)}:
        raise RuntimeError(
            f"{subject} {dataset_name}: unexpected subject values."
        )

    session_values = set(metadata["session"].astype(str))
    if session_values != set(SESSION_ORDER):
        raise RuntimeError(
            f"{subject} {dataset_name}: expected both S1 and S2, "
            f"found {sorted(session_values)}."
        )

    duplicated = metadata.duplicated(
        [
            "subject",
            "session",
            "cv_block",
            "trial_within_block",
            "identity",
            "identity_repetition_within_block",
        ],
        keep=False,
    )
    if duplicated.any():
        bad = metadata.loc[
            duplicated,
            [
                "subject",
                "session",
                "cv_block",
                "trial_within_block",
                "identity",
                "identity_repetition_within_block",
            ],
        ].head(20)
        raise RuntimeError(
            f"{subject} {dataset_name}: duplicate single-trial keys:\n"
            + bad.to_string(index=False)
        )


# =============================================================================
# LOAD, CONCATENATE, AND SORT SESSIONS
# =============================================================================


def load_and_combine_sessions(
    preprocessed_root: Path,
    subject: str,
    dataset_name: str,
    preload: bool = True,
    verbose: str | bool | None = "INFO",
) -> tuple[mne.Epochs, list[Path]]:
    """
    Load and combine all available final-epoch sessions.

    Complete subjects
    -----------------
    When both S1 and S2 are available, they are loaded and concatenated
    in the original S1-then-S2 order.

    Incomplete subjects
    -------------------
    When only one session is available, that session is retained without
    creating an artificial replacement session.

    Missing experimental blocks are allowed. Their original cv_block values
    are preserved from the Epochs metadata and are never renumbered or
    zero-padded.
    """

    session_epochs: list[mne.Epochs] = []
    source_paths: list[Path] = []

    available_sessions: list[str] = []
    missing_sessions: list[str] = []

    for session in ("S1", "S2"):

        source_path = build_session_path(
            preprocessed_root=preprocessed_root,
            subject=subject,
            session=session,
            dataset_name=dataset_name,
        )

        if not source_path.is_file():

            missing_sessions.append(
                session
            )

            print(
                f"[INFO] {subject} {dataset_name}: "
                f"{session} final Epochs file is unavailable; "
                f"skipping {source_path}"
            )

            continue

        print(
            f"[LOAD] {source_path}"
        )

        epochs = mne.read_epochs(
            source_path,
            preload=preload,
            verbose=verbose,
        )

        validate_one_session_epochs(
            epochs=epochs,
            subject=subject,
            session=session,
            dataset_name=dataset_name,
            source_path=source_path,
        )

        session_epochs.append(
            epochs
        )

        source_paths.append(
            source_path.resolve()
        )

        available_sessions.append(
            session
        )

    if not session_epochs:

        checked_paths = [
            build_session_path(
                preprocessed_root=preprocessed_root,
                subject=subject,
                session=session,
                dataset_name=dataset_name,
            )
            for session in ("S1", "S2")
        ]

        raise FileNotFoundError(
            f"{subject} {dataset_name}: no final Epochs files "
            f"were available. Checked: "
            f"{[str(path) for path in checked_paths]}"
        )

    # -------------------------------------------------------------------------
    # Validate compatibility whenever multiple sessions exist.
    # -------------------------------------------------------------------------

    reference_epochs = session_epochs[0]
    reference_session = available_sessions[0]

    for current_session, current_epochs in zip(
        available_sessions[1:],
        session_epochs[1:],
    ):

        if (
            reference_epochs.ch_names
            != current_epochs.ch_names
        ):

            raise RuntimeError(
                f"{subject} {dataset_name}: channel names or channel "
                f"order differ between {reference_session} and "
                f"{current_session}."
            )

        if not np.array_equal(
            reference_epochs.times,
            current_epochs.times,
        ):

            raise RuntimeError(
                f"{subject} {dataset_name}: epoch time vectors differ "
                f"between {reference_session} and {current_session}."
            )

        if not np.isclose(
            float(reference_epochs.info["sfreq"]),
            float(current_epochs.info["sfreq"]),
        ):

            raise RuntimeError(
                f"{subject} {dataset_name}: sampling rates differ "
                f"between {reference_session} and {current_session}."
            )

    # -------------------------------------------------------------------------
    # Combine sessions.
    #
    # For complete subjects, this preserves the original S1-then-S2
    # concatenation. For one-session subjects, no fake session is introduced.
    # -------------------------------------------------------------------------

    if len(session_epochs) == 1:

        combined = session_epochs[0].copy()

    else:

        combined = mne.concatenate_epochs(
            session_epochs,
            add_offset=True,
            on_mismatch="raise",
            verbose=verbose,
        )

    if combined.metadata is None:

        raise RuntimeError(
            f"{subject} {dataset_name}: metadata disappeared after "
            "loading/concatenating available sessions."
        )

    combined.metadata = (
        combined.metadata
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Stable observation ordering.
    #
    # This maintains:
    # S1 before S2
    # lower cv_block before higher cv_block
    # original trial order within block
    #
    # Missing cv_block values remain missing and are never renumbered.
    # -------------------------------------------------------------------------

    required_order_columns = [
        "session_number",
        "cv_block",
        "trial_within_block",
        "identity",
        "identity_repetition_within_block",
    ]

    missing_order_columns = [
        column
        for column in required_order_columns
        if column not in combined.metadata.columns
    ]

    if missing_order_columns:

        raise RuntimeError(
            f"{subject} {dataset_name}: metadata is missing columns "
            f"needed for stable sorting: "
            f"{missing_order_columns}"
        )

    metadata_for_sort = (
        combined.metadata.copy()
    )

    metadata_for_sort[
        "_epoch_position"
    ] = np.arange(
        len(metadata_for_sort),
        dtype=int,
    )

    order = (
        metadata_for_sort
        .sort_values(
            required_order_columns,
            kind="stable",
        )["_epoch_position"]
        .to_numpy(dtype=int)
    )

    if len(order) != len(combined):

        raise RuntimeError(
            f"{subject} {dataset_name}: sort order length "
            f"{len(order)} does not match epoch count "
            f"{len(combined)}."
        )

    if len(order) > 0:

        if (
            order.min() < 0
            or order.max() >= len(combined)
        ):

            raise RuntimeError(
                f"{subject} {dataset_name}: invalid epoch sort "
                f"positions {order.min()}-{order.max()} for "
                f"{len(combined)} epochs."
            )

    combined = combined[
        order
    ]

    combined.metadata = (
        combined.metadata
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Audit available sessions and blocks.
    # -------------------------------------------------------------------------

    observed_sessions = (
        combined.metadata[
            "session"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    observed_cv_blocks = sorted(
        combined.metadata[
            "cv_block"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    block_mapping_columns = [
        "session",
        "session_number",
        "block_code",
        "block_within_session",
        "cv_block",
    ]

    block_mapping = (
        combined.metadata[
            block_mapping_columns
        ]
        .drop_duplicates()
        .sort_values(
            [
                "session_number",
                "cv_block",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(observed_cv_blocks) < 2:

        raise RuntimeError(
            f"{subject} {dataset_name}: only "
            f"{len(observed_cv_blocks)} unique cv_block value(s) "
            "remain. At least two blocks are required for "
            "leave-one-block-out cross-validation."
        )

    print(
        f"[COMBINE] {subject} {dataset_name}: "
        f"{len(combined)} valid single-trial epochs."
    )

    print(
        f"[SESSIONS] available={observed_sessions}; "
        f"missing={missing_sessions}"
    )

    print(
        f"[CV BLOCKS] {observed_cv_blocks}"
    )

    print(
        "[BLOCK MAPPING]\n"
        + block_mapping.to_string(
            index=False
        )
    )

    return (
        combined,
        source_paths,
    )

# =============================================================================
# PAPER-MATCHED ANALYSIS TIME SELECTION
# =============================================================================


def find_paper_time_indices(
    epochs: mne.BaseEpochs,
    *,
    start_seconds: float = ANALYSIS_WINDOW_START_SECONDS,
    n_timepoints: int = ANALYSIS_N_TIMEPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly 307 consecutive samples beginning nearest 50 ms."""

    times = np.asarray(epochs.times, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise RuntimeError(
            f"Invalid Epochs time vector with shape {times.shape}."
        )
    if not np.isfinite(times).all():
        raise RuntimeError("Epochs time vector contains NaN or infinity.")

    sfreq = float(epochs.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise RuntimeError(
            f"Expected {EXPECTED_SFREQ} Hz, got {sfreq} Hz."
        )

    start_seconds = float(start_seconds)
    n_timepoints = int(n_timepoints)
    if n_timepoints <= 0:
        raise ValueError(
            f"n_timepoints must be positive, got {n_timepoints}."
        )

    start_index = int(np.argmin(np.abs(times - start_seconds)))
    stop_index = start_index + n_timepoints

    if stop_index > len(times):
        raise RuntimeError(
            f"The requested {n_timepoints}-sample window exceeds the epoch "
            f"length. Epoch contains {len(times)} samples; start index is "
            f"{start_index}."
        )

    indices = np.arange(start_index, stop_index, dtype=int)
    selected_times = times[indices]

    one_sample = 1.0 / sfreq
    if abs(selected_times[0] - start_seconds) > one_sample:
        raise RuntimeError(
            f"Selected first sample ({selected_times[0]:.9f} s) is not "
            f"within one sample of {start_seconds:.9f} s."
        )

    expected_end = start_seconds + (n_timepoints - 1) / sfreq
    if abs(selected_times[-1] - expected_end) > one_sample:
        raise RuntimeError(
            f"Selected final sample ({selected_times[-1]:.9f} s) is not "
            f"within one sample of expected end {expected_end:.9f} s."
        )

    return indices, selected_times


find_analysis_time_indices = find_paper_time_indices


# =============================================================================
# GROUPING AND REPETITION QC
# =============================================================================


def build_identity_block_groups(
    single_metadata: pd.DataFrame,
) -> list[tuple[tuple[Any, ...], np.ndarray]]:
    """Return stable (subject, cv_block, identity) groups and row positions."""

    missing = set(BLOCK_GROUP_COLUMNS) - set(single_metadata.columns)
    if missing:
        raise ValueError(
            f"Cannot build block groups; missing columns: {sorted(missing)}"
        )

    metadata = single_metadata.reset_index(drop=True)
    grouped = metadata.groupby(
        BLOCK_GROUP_COLUMNS,
        sort=True,
        dropna=False,
    )

    groups: list[tuple[tuple[Any, ...], np.ndarray]] = []
    for group_key, row_positions in grouped.indices.items():
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        positions = np.asarray(row_positions, dtype=int)
        if positions.ndim != 1 or len(positions) == 0:
            raise RuntimeError(
                f"Invalid positions for group {group_key}: {positions}"
            )
        groups.append((tuple(group_key), positions))

    if not groups:
        raise RuntimeError(
            "No subject/cv_block/identity groups were available."
        )

    return groups


def validate_group_constant_columns(
    rows: pd.DataFrame,
    group_key: tuple[Any, ...],
    *,
    constant_columns: Sequence[str] = tuple(BLOCK_CONSTANT_COLUMNS),
) -> None:
    """Ensure expected metadata are constant within an averaging group."""

    missing = set(constant_columns) - set(rows.columns)
    if missing:
        raise ValueError(
            f"Group {group_key}: missing constant columns {sorted(missing)}"
        )

    for column in constant_columns:
        if rows[column].nunique(dropna=False) != 1:
            values = rows[column].drop_duplicates().tolist()
            raise RuntimeError(
                f"Group {group_key}: metadata column {column!r} is not "
                f"constant. Values: {values}"
            )


def summarize_repetition_counts(
    single_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return valid repetition counts per session/block/identity."""

    required = {"session", "cv_block", "block_code", "identity"}
    missing = required - set(single_metadata.columns)
    if missing:
        raise ValueError(
            f"Cannot summarize repetition counts; missing: {sorted(missing)}"
        )

    return (
        single_metadata
        .groupby(
            ["session", "cv_block", "block_code", "identity"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )


def print_abnormal_repetition_counts(
    single_metadata: pd.DataFrame,
    *,
    expected_count: int = 4,
) -> pd.DataFrame:
    """Print and return groups with unexpected valid repetition counts."""

    counts = summarize_repetition_counts(single_metadata)
    abnormal = counts[counts["count"] != int(expected_count)].copy()

    print(f"\n[DEBUG] Groups with repetition count != {expected_count}:")
    if abnormal.empty:
        print("None")
        return abnormal

    print(abnormal.to_string(index=False))

    for row in abnormal.itertuples(index=False):
        matching = single_metadata[
            (single_metadata["session"].astype(str) == str(row.session))
            & (single_metadata["cv_block"].astype(int) == int(row.cv_block))
            & (single_metadata["identity"].astype(int) == int(row.identity))
        ].copy()

        columns_to_show = [
            column
            for column in [
                "subject",
                "session",
                "session_number",
                "block_code",
                "block_within_session",
                "cv_block",
                "trial_within_block",
                "identity",
                "original_event_code",
                "identity_repetition_within_block",
                "event_sample",
            ]
            if column in matching.columns
        ]

        sort_columns = [
            column
            for column in [
                "trial_within_block",
                "identity_repetition_within_block",
            ]
            if column in columns_to_show
        ]

        print(
            f"\n[DEBUG DETAILS] session={row.session}, "
            f"cv_block={row.cv_block}, block_code={row.block_code}, "
            f"identity={row.identity}, count={row.count}"
        )

        if sort_columns:
            matching = matching.sort_values(sort_columns, kind="stable")

        print(matching[columns_to_show].to_string(index=False))

    return abnormal


# =============================================================================
# GENERIC WITHIN-BLOCK AVERAGING
# =============================================================================


def validate_trial_first_array(
    data: np.ndarray,
    metadata: pd.DataFrame,
    *,
    array_name: str = "data",
) -> None:
    """Validate an array whose first axis indexes observations."""

    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"{array_name} must be a NumPy array, "
            f"got {type(data).__name__}."
        )
    if data.ndim < 1:
        raise ValueError(f"{array_name} must have at least one dimension.")
    if data.shape[0] != len(metadata):
        raise RuntimeError(
            f"{array_name} trial count ({data.shape[0]}) does not match "
            f"metadata rows ({len(metadata)})."
        )
    if not np.isfinite(data).all():
        raise RuntimeError(f"{array_name} contains NaN or infinity.")


def average_trial_first_array(
    single_data: np.ndarray,
    single_metadata: pd.DataFrame,
    *,
    output_dtype: np.dtype | type | None = np.float32,
    mean_dtype: np.dtype | type | None = np.float64,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Average any trial-first array within subject/cv_block/identity.

    Use directly for raw and power. For phase, do not average angles; first
    convert them to unit complex vectors, then call this function with
    output_dtype=np.complex64 and mean_dtype=np.complex128.
    """

    metadata = single_metadata.reset_index(drop=True).copy()
    validate_trial_first_array(single_data, metadata, array_name="single_data")
    groups = build_identity_block_groups(metadata)

    averaged_arrays: list[np.ndarray] = []
    averaged_rows: list[dict[str, Any]] = []

    for group_key, positions in groups:
        rows = metadata.iloc[positions]
        validate_group_constant_columns(rows, group_key)

        group_data = single_data[positions]
        effective_mean_dtype = mean_dtype
        if np.iscomplexobj(group_data) and mean_dtype is np.float64:
            effective_mean_dtype = np.complex128

        averaged_arrays.append(
            group_data.mean(axis=0, dtype=effective_mean_dtype)
        )

        first = rows.iloc[0]
        constituent_ids = (
            rows["observation_id"].astype(str).tolist()
            if "observation_id" in rows.columns
            else []
        )
        repetitions = sorted(
            rows["identity_repetition_within_block"].astype(int).tolist()
        )

        averaged_rows.append(
            {
                "subject": str(first["subject"]),
                "session": str(first["session"]),
                "session_number": int(first["session_number"]),
                "block_code": int(first["block_code"]),
                "block_within_session": int(first["block_within_session"]),
                "cv_block": int(first["cv_block"]),
                "identity": int(first["identity"]),
                "race": str(first["race"]),
                "original_event_code": int(first["original_event_code"]),
                "is_oddball": bool(first["is_oddball"]),
                "trial_mode": "block_average",
                "n_trials_averaged": int(len(positions)),
                "available_repetitions": ",".join(map(str, repetitions)),
                "constituent_observation_ids": "|".join(constituent_ids),
            }
        )

    if not averaged_arrays:
        raise RuntimeError(
            "No identity-by-block groups were available to average."
        )

    averaged_data = np.stack(averaged_arrays)
    if output_dtype is not None:
        averaged_data = averaged_data.astype(output_dtype, copy=False)

    averaged_metadata = pd.DataFrame(averaged_rows)

    original_keys = [
        (str(row["subject"]), int(row["cv_block"]), int(row["identity"]))
        for row in averaged_rows
    ]
    if len(original_keys) != len(set(original_keys)):
        raise RuntimeError(
            "Averaging generated duplicate subject/cv_block/identity keys."
        )

    key_to_array = {
        key: array for key, array in zip(original_keys, averaged_data)
    }

    averaged_metadata = averaged_metadata.sort_values(
        BLOCK_SORT_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)

    averaged_data = np.stack(
        [
            key_to_array[
                (str(row.subject), int(row.cv_block), int(row.identity))
            ]
            for row in averaged_metadata.itertuples(index=False)
        ]
    )

    if output_dtype is not None:
        averaged_data = averaged_data.astype(output_dtype, copy=False)

    if "observation_id" in averaged_metadata.columns:
        averaged_metadata = averaged_metadata.drop(columns=["observation_id"])

    averaged_metadata.insert(
        0,
        "observation_id",
        make_trial_ids(averaged_metadata, mode="block_average"),
    )

    if averaged_metadata.duplicated(BLOCK_GROUP_COLUMNS).any():
        raise RuntimeError(
            "Block-averaged output contains duplicate "
            "subject/cv_block/identity rows."
        )

    if not averaged_metadata["n_trials_averaged"].between(1, 4).all():
        bad = averaged_metadata.loc[
            ~averaged_metadata["n_trials_averaged"].between(1, 4),
            ["cv_block", "identity", "n_trials_averaged"],
        ]
        raise RuntimeError(
            "Unexpected repetition counts after block averaging:\n"
            + bad.to_string(index=False)
        )

    if not np.isfinite(averaged_data).all():
        raise RuntimeError("Block-averaged data contain NaN or infinity.")

    return averaged_data, averaged_metadata


average_real_within_identity_block = average_trial_first_array
average_complex_vectors_within_identity_block = average_trial_first_array


# =============================================================================
# STANDARD FEATURE AXIS CONVERSION
# =============================================================================


def trial_channel_freq_time_to_standard(
    data: np.ndarray,
    *,
    dtype: np.dtype | type | None = np.float32,
) -> np.ndarray:
    """Convert (trial, channel, frequency, time) to (freq,time,trial,channel)."""

    if data.ndim != 4:
        raise ValueError(
            "Expected (trial, channel, frequency, time), "
            f"got {data.shape}."
        )

    feature = np.transpose(data, (2, 3, 0, 1))
    if dtype is not None:
        feature = feature.astype(dtype, copy=False)
    return feature


def trial_channel_time_to_standard_raw(
    data: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Convert raw (trial, channel, time) to (1,time,trial,channel)."""

    if data.ndim != 3:
        raise ValueError(
            f"Expected (trial, channel, time), got {data.shape}."
        )

    feature = np.transpose(data, (2, 0, 1))[np.newaxis, ...]
    return feature.astype(dtype, copy=False)


# =============================================================================
# CROSS-DATASET CONSISTENCY
# =============================================================================


def validate_all_a_c_consistency(
    outputs: dict[str, dict[str, pd.DataFrame]],
    *,
    modes: Iterable[str] = ("single", "block_average"),
) -> None:
    """Confirm all_id observation keys equal the union of a_id and c_id."""

    missing_datasets = set(DATASET_ORDER) - set(outputs)
    if missing_datasets:
        raise ValueError(
            f"Missing dataset outputs: {sorted(missing_datasets)}"
        )

    key_columns_by_mode = {
        "single": [
            "subject",
            "session",
            "cv_block",
            "trial_within_block",
            "identity",
            "identity_repetition_within_block",
        ],
        "block_average": ["subject", "cv_block", "identity"],
    }

    for mode in modes:
        if mode not in key_columns_by_mode:
            raise ValueError(f"Unknown consistency mode {mode!r}.")

        key_columns = key_columns_by_mode[mode]

        for dataset_name in DATASET_ORDER:
            if mode not in outputs[dataset_name]:
                raise ValueError(
                    f"Missing {mode!r} metadata for {dataset_name!r}."
                )

        all_keys = set(
            map(
                tuple,
                outputs["all_id"][mode][key_columns]
                .astype(str)
                .to_numpy(),
            )
        )

        race_union = pd.concat(
            [outputs["a_id"][mode], outputs["c_id"][mode]],
            ignore_index=True,
        )
        race_keys = set(
            map(tuple, race_union[key_columns].astype(str).to_numpy())
        )

        if all_keys != race_keys:
            missing_from_union = list(all_keys - race_keys)[:10]
            extra_in_union = list(race_keys - all_keys)[:10]
            raise RuntimeError(
                f"all_id does not match a_id + c_id for {mode}. "
                f"Missing from union: {missing_from_union}; "
                f"extra in union: {extra_in_union}."
            )

    print("[VALIDATION] all_id equals the union of a_id and c_id.")


# =============================================================================
# SHARED AUDIT METADATA
# =============================================================================


def build_shared_eeg_audit_metadata(
    *,
    selected_times: Sequence[float],
    source_paths: Sequence[Path],
    dataset_name: str,
    observation_mode: str,
) -> dict[str, Any]:
    """Build audit fields shared by raw, power, and phase metadata."""

    dataset_name = validate_dataset_name(dataset_name)
    if observation_mode not in {"single", "block_average"}:
        raise ValueError(
            f"Invalid observation_mode {observation_mode!r}."
        )

    selected_times_array = np.asarray(selected_times, dtype=np.float64)
    if selected_times_array.shape != (ANALYSIS_N_TIMEPOINTS,):
        raise ValueError(
            f"Expected {ANALYSIS_N_TIMEPOINTS} selected times, "
            f"got shape {selected_times_array.shape}."
        )

    return {
        "dataset_name": dataset_name,
        "observation_mode": observation_mode,
        "source_epoch_files": [
            str(Path(path).resolve()) for path in source_paths
        ],
        "input_filter_band_hz": [0.1, 40.0],
        "sampling_rate_hz": EXPECTED_SFREQ,
        "requested_time_window_seconds": [0.050, 0.650],
        "selection_rule": (
            "307 consecutive samples beginning at the epoch sample nearest "
            "0.050 seconds; wavelet features are convolved on the complete "
            "epoch before the identical post-TFR selection"
        ),
        "selected_time_start_seconds": float(selected_times_array[0]),
        "selected_time_end_seconds": float(selected_times_array[-1]),
        "n_timepoints": int(len(selected_times_array)),
        "channel_selection": "Shoura et al. occipitotemporal ROI",
        "channel_order": list(OT_CHANNELS),
        "n_channels": len(OT_CHANNELS),
        "z_scored": False,
        "z_score_stage": "SVM cross-validation training fold",
        "zero_padding_used": False,
    }
