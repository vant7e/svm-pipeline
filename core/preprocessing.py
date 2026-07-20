#!/usr/bin/env python3
"""
EEG preprocessing pipeline reproducing the Letswave protocol.
"""

from __future__ import annotations

from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.signal import detrend as scipy_detrend

import os

SUBJECT = os.environ["SUBJECT"]
SESSION = os.environ["SESSION"]

print(f"Processing: {SUBJECT} {SESSION}")

BDF_DIRECTORY = Path(
    "MAIN_PATH_FOR_YOUR_PROJECT"
)

if not BDF_DIRECTORY.is_dir():
    raise NotADirectoryError(
        f"BDF directory does not exist or is not a directory: "
        f"{BDF_DIRECTORY}"
    )

def import_bdf_files(
    bdf_files: list[Path],
    status_channel: str = "Status",
) -> mne.io.BaseRaw:
    """
    Import one or more consecutive BioSemi BDF files.

    When multiple BDF files belong to the same session, they are loaded
    in acquisition order and concatenated into one continuous Raw object.
    """

    if not bdf_files:
        raise FileNotFoundError(
            "No BDF files were supplied."
        )

    raw_parts = []

    for part_number, bdf_file in enumerate(
        bdf_files,
        start=1,
    ):
        if not bdf_file.is_file():
            raise FileNotFoundError(
                f"BDF file not found: {bdf_file}"
            )

        print(
            f"Importing BDF part {part_number}/"
            f"{len(bdf_files)}: {bdf_file}"
        )

        part_raw = mne.io.read_raw_bdf(
            str(bdf_file),
            preload=True,
            stim_channel=status_channel,
            infer_types=False,
            verbose=True,
        )

        raw_parts.append(
            part_raw
        )

    # Validate that all parts are compatible before concatenation.
    reference_channels = raw_parts[0].ch_names
    reference_sfreq = float(
        raw_parts[0].info["sfreq"]
    )

    for part_number, part_raw in enumerate(
        raw_parts[1:],
        start=2,
    ):
        if part_raw.ch_names != reference_channels:
            raise RuntimeError(
                f"BDF part {part_number} has different "
                "channel names or channel order."
            )

        if not np.isclose(
            part_raw.info["sfreq"],
            reference_sfreq,
        ):
            raise RuntimeError(
                f"BDF part {part_number} has sampling rate "
                f"{part_raw.info['sfreq']} Hz, but the first "
                f"part has {reference_sfreq} Hz."
            )

    if len(raw_parts) == 1:
        raw = raw_parts[0]

    else:
        raw = mne.concatenate_raws(
            raw_parts,
            preload=True,
            verbose="INFO",
        )
        
    print("Imported files:")
    for f in bdf_files:
        print(f"  {f.name}")

    print("===================================")
    print("STEP 1: Import BDF")
    print("===================================")
    print(f"BDF parts     : {len(bdf_files)}")
    print(f"Sampling rate : {raw.info['sfreq']} Hz")
    print(f"Channels      : {len(raw.ch_names)}")
    print(f"Samples       : {raw.n_times}")
    print(
        f"Duration      : "
        f"{raw.n_times / raw.info['sfreq']:.3f} seconds"
    )

    return raw

def assign_electrode_coordinates(
    raw: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    """
    STEP 2

    Assign BioSemi64 electrode coordinates.
    """

    raw = raw.copy()

    montage = mne.channels.make_standard_montage(
        "biosemi64"
    )

    raw.set_montage(
        montage,
        match_case=False,
        on_missing="warn",
    )

    print("STEP 2: BioSemi64 montage assigned")

    return raw

def extract_events(
    raw: mne.io.BaseRaw,
    status_channel: str = "Status",
    bitshift: int = 0,
) -> np.ndarray:
    """
    Read triggers from the BioSemi Status channel.

    Only the lowest 8 bits are retained, corresponding to trigger
    values 0-255 used by this experiment.
    """

    events = mne.find_events(
        raw,
        stim_channel=status_channel,
        shortest_event=1,
        consecutive=True,
        uint_cast=True,
        mask=0xFF,
        mask_type="and",
        initial_event=False,
        verbose=True,
    ).astype(int)

    if bitshift:
        events[:, 2] >>= bitshift

    print(
        f"Found {len(events)} masked events."
    )

    print(
        "Masked event IDs: "
        f"{np.unique(events[:, 2]).tolist()}"
    )

    return events

def remove_events_near_concatenation_boundaries(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    guard_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove triggers occurring close to BDF concatenation boundaries.

    This protects against corrupted Status-channel values at the end or
    beginning of split BDF recordings.

    Parameters
    ----------
    raw
        Concatenated Raw object containing BAD/EDGE boundary annotations.

    events
        Events extracted from the Status channel.

    guard_seconds
        Remove events within this many seconds before or after each boundary.

    Returns
    -------
    retained_events
        Events outside all boundary guard intervals.

    removed_events
        Events removed because they were close to a boundary.
    """

    sfreq = float(raw.info["sfreq"])

    boundary_samples = []

    for onset, description in zip(
        raw.annotations.onset,
        raw.annotations.description,
    ):
        description_lower = str(description).lower()

        if (
            "boundary" in description_lower
            or description_lower.startswith("bad boundary")
            or description_lower.startswith("edge boundary")
        ):
            boundary_sample = (
                int(round(float(onset) * sfreq))
                + int(raw.first_samp)
            )

            boundary_samples.append(
                boundary_sample
            )

    boundary_samples = sorted(
        set(boundary_samples)
    )

    if not boundary_samples:
        print(
            "No BDF concatenation boundaries were found; "
            "no boundary events were removed."
        )

        return events.copy(), np.empty(
            (0, 3),
            dtype=int,
        )

    guard_samples = int(
        round(
            guard_seconds * sfreq
        )
    )

    remove_mask = np.zeros(
        len(events),
        dtype=bool,
    )

    print(
        "\nConcatenation boundary trigger screening:"
    )

    for boundary_sample in boundary_samples:

        distance = np.abs(
            events[:, 0]
            - boundary_sample
        )

        near_boundary = (
            distance <= guard_samples
        )

        remove_mask |= near_boundary

        nearby_events = events[
            near_boundary
        ]

        print(
            f"Boundary sample {boundary_sample}: "
            f"removing {len(nearby_events)} event(s) within "
            f"±{guard_seconds:.3f} seconds."
        )

        if len(nearby_events) > 0:

            print(
                "  Nearby samples: "
                f"{nearby_events[:20, 0].tolist()}"
            )

            print(
                "  Nearby codes: "
                f"{nearby_events[:20, 2].tolist()}"
            )

    retained_events = events[
        ~remove_mask
    ].copy()

    removed_events = events[
        remove_mask
    ].copy()

    print(
        f"Removed {len(removed_events)} boundary-adjacent "
        f"event(s); retained {len(retained_events)}."
    )

    return retained_events, removed_events


UNWANTED_EVENTS = [
    251,
    253,
    254,
    255,
    80,
]


def remove_unwanted_events(
    events: np.ndarray,
):
    """
    Remove unwanted trigger codes.

    Keep 252.
    """

    mask = np.isin(
        events[:, 2],
        UNWANTED_EVENTS,
    )

    retained = events[~mask]

    removed = events[mask]

    print(
        f"Removed {len(removed)} events."
    )

    return retained, removed

def events_to_annotations(
    raw: mne.io.BaseRaw,
    retained_events: np.ndarray,
) -> mne.io.BaseRaw:
    """
    Store retained triggers as annotations before removing Status.

    Existing non-numeric annotations are preserved.
    Existing numeric annotations are replaced to avoid duplicate triggers.
    """

    mapping = {
        int(code): str(int(code))
        for code in np.unique(
            retained_events[:, 2]
        )
    }

    trigger_annotations = mne.annotations_from_events(
        retained_events,
        sfreq=raw.info["sfreq"],
        event_desc=mapping,
        first_samp=raw.first_samp,
        orig_time=raw.info["meas_date"],
    )

    output = raw.copy()

    non_numeric_annotations = output.annotations[
        [
            not str(description).isdigit()
            for description
            in output.annotations.description
        ]
    ]

    output.set_annotations(
        non_numeric_annotations
        + trigger_annotations
    )

    print(
        f"Stored {len(trigger_annotations)} numeric trigger annotations."
    )

    return output


def remove_channels(
    raw,
    status_channel="Status",
):
    """
    Remove

    EX1-EX8

    EXG1-EXG8

    Status
    """

    raw = raw.copy()

    pattern = re.compile(
        r"^(EX|EXG)[1-8]$",
        re.I,
    )

    drop = []

    for ch in raw.ch_names:

        if pattern.fullmatch(ch):

            drop.append(ch)

    drop.append(status_channel)

    raw.drop_channels(drop)

    print("Removed")

    print(drop)

    return raw

def butterworth_filter(
    raw,
    n_jobs=1,
):
    """
    Butterworth

    0.1-40 Hz
    """

    raw = raw.copy()

    raw.filter(
        l_freq=0.1,
        h_freq=40,
        picks="eeg",
        method="iir",
        iir_params=dict(
            order=4,
            ftype="butter",
        ),
        phase="zero",
        n_jobs=n_jobs,
    )

    return raw

TEST_BLOCKS = range(
    202,
    218,
)


def events_from_annotations_keep_codes(
    raw: mne.io.BaseRaw,
) -> np.ndarray:
    """
    Convert numeric trigger annotations back to events while preserving
    the original trigger numbers.

    For example:
        annotation "202" -> event code 202
        annotation "1"   -> event code 1
    """

    descriptions = sorted(
        {
            str(annotation["description"])
            for annotation in raw.annotations
            if str(annotation["description"]).isdigit()
        },
        key=int,
    )

    if not descriptions:
        raise RuntimeError(
            "No numeric trigger annotations were found."
        )

    event_id = {
        description: int(description)
        for description in descriptions
    }

    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        regexp=None,
        verbose="ERROR",
    )

    return events.astype(int)


def find_block_events(
    raw: mne.io.BaseRaw,
) -> dict[int, int]:
    """
    Find exactly one valid testing block trigger for each code 202-217.
    """

    events = events_from_annotations_keep_codes(
        raw
    )

    blocks = {}

    print("\nTesting block trigger locations:")

    for code in TEST_BLOCKS:

        samples = events[
            events[:, 2] == code,
            0,
        ]

        print(
            f"  Block trigger {code}: "
            f"{samples.astype(int).tolist()}"
        )

        if len(samples) != 1:

            raise RuntimeError(
                f"Expected exactly one valid block trigger {code}, "
                f"but found {len(samples)} at samples "
                f"{samples.astype(int).tolist()}."
            )

        blocks[code] = int(
            samples[0]
        )

    return blocks

BLOCK_START = -0.02

EXPECTED_SFREQ = 512.0

def calculate_minimum_block_length(
    block_events: dict[int, int],
    sfreq: float,
) -> tuple[int, float]:
    """
    Calculate the minimum interval between consecutive testing-block
    triggers.

    For example:
        block 202 length = trigger 203 - trigger 202
        block 203 length = trigger 204 - trigger 203

    The minimum interval is then used as the common segmentation length
    for all testing blocks.
    """

    ordered_codes = list(TEST_BLOCKS)

    interval_rows = []

    for current_code, next_code in zip(
        ordered_codes[:-1],
        ordered_codes[1:],
    ):
        current_sample = int(
            block_events[current_code]
        )

        next_sample = int(
            block_events[next_code]
        )

        interval_samples = (
            next_sample
            - current_sample
        )

        if interval_samples <= 0:
            raise RuntimeError(
                f"Invalid block-trigger order: "
                f"block {current_code} occurs at sample "
                f"{current_sample}, while block {next_code} "
                f"occurs at sample {next_sample}."
            )

        interval_seconds = (
            interval_samples / sfreq
        )

        interval_rows.append(
            {
                "current_code": current_code,
                "next_code": next_code,
                "samples": interval_samples,
                "seconds": interval_seconds,
            }
        )

    print(
        "\nConsecutive testing-block intervals:"
    )

    for row in interval_rows:
        print(
            f"  {row['current_code']} -> "
            f"{row['next_code']}: "
            f"{row['samples']} samples, "
            f"{row['seconds']:.6f} seconds"
        )

    shortest_row = min(
        interval_rows,
        key=lambda row: row["samples"],
    )

    minimum_samples = int(
        shortest_row["samples"]
    )

    minimum_seconds = float(
        shortest_row["seconds"]
    )

    print(
        "\nShortest testing-block interval:"
    )

    print(
        f"  {shortest_row['current_code']} -> "
        f"{shortest_row['next_code']}"
    )

    print(
        f"  {minimum_samples} samples"
    )

    print(
        f"  {minimum_seconds:.6f} seconds"
    )

    return (
        minimum_samples,
        minimum_seconds,
    )
    
    

def segment_blocks(
    raw: mne.io.BaseRaw,
    block_events: dict[int, int],
) -> dict[int, mne.io.BaseRaw]:
    """
    Segment all testing blocks using the minimum interval between
    consecutive testing-block triggers.

    Each block begins BLOCK_START seconds relative to its block trigger.
    All blocks receive the same number of samples.

    For the final block, all available data are retained if the recording
    ends before the requested common block length.
    """

    sfreq = float(
        raw.info["sfreq"]
    )

    raw_first_sample = int(
        raw.first_samp
    )

    raw_last_sample = (
        raw_first_sample
        + int(raw.n_times)
        - 1
    )

    pre_samples = int(
        round(
            BLOCK_START
            * sfreq
        )
    )

    (
        requested_samples,
        requested_duration_seconds,
    ) = calculate_minimum_block_length(
        block_events=block_events,
        sfreq=sfreq,
    )

    print(
        "\nCommon block segmentation length:"
    )

    print(
        f"  {requested_samples} samples"
    )

    print(
        f"  {requested_duration_seconds:.6f} seconds"
    )

    blocks = {}

    for code in TEST_BLOCKS:

        event_sample = int(
            block_events[code]
        )

        start_sample = (
            event_sample
            + pre_samples
        )

        requested_stop_sample = (
            start_sample
            + requested_samples
            - 1
        )

        actual_stop_sample = min(
            requested_stop_sample,
            raw_last_sample,
        )

        available_samples = (
            actual_stop_sample
            - start_sample
            + 1
        )

        if available_samples <= 0:
            raise RuntimeError(
                f"Block {code}: no data are available "
                "after the block trigger."
            )

        missing_samples = (
            requested_samples
            - available_samples
        )

        if missing_samples > 0:
            print(
                f"[WARNING] Block {code}: requested "
                f"{requested_samples} samples "
                f"({requested_duration_seconds:.6f} seconds), "
                f"but the recording ends "
                f"{missing_samples} samples "
                f"({missing_samples / sfreq:.6f} seconds) early. "
                "Using all available data."
            )

        tmin = (
            start_sample
            - raw_first_sample
        ) / sfreq

        tmax = (
            actual_stop_sample
            - raw_first_sample
        ) / sfreq

        block = raw.copy().crop(
            tmin=tmin,
            tmax=tmax,
            include_tmax=True,
        )

        if block.n_times != available_samples:
            raise RuntimeError(
                f"Block {code}: expected "
                f"{available_samples} available samples, "
                f"but crop produced "
                f"{block.n_times}."
            )

        print(
            f"Block {code}: "
            f"{block.n_times} samples, "
            f"{block.n_times / sfreq:.6f} seconds; "
            f"samples {start_sample}-"
            f"{actual_stop_sample}."
        )

        blocks[code] = block

    return blocks

# =============================================================================
# STEP 11 — VALIDATE EVENTS INSIDE EACH BLOCK
#
# General experiment structure:
# 60 normal face identities × 4 repetitions = 240 normal trials
# 24 oddball trials
#
# Asian normal stimuli:       1-30
# Caucasian normal stimuli:  31-60
# Asian oddballs:           101-130
# Caucasian oddballs:       131-160
# =============================================================================

STIMULUS_CODES = tuple(range(1, 61))

ASIAN_STIMULUS_CODES = tuple(range(1, 31))

CAUCASIAN_STIMULUS_CODES = tuple(range(31, 61))

ASIAN_ODDBALL_CODES = tuple(range(101, 131))

CAUCASIAN_ODDBALL_CODES = tuple(range(131, 161))

ODDBALL_CODES = (
    ASIAN_ODDBALL_CODES
    + CAUCASIAN_ODDBALL_CODES
)

EXPECTED_NORMAL_TRIALS_PER_BLOCK = 240

EXPECTED_ODDBALL_TRIALS_PER_BLOCK = 24

EXPECTED_REPETITIONS_PER_IDENTITY = 4


def validate_block_events(
    block: mne.io.BaseRaw,
    block_code: int,
) -> pd.DataFrame:
    """
    Inspect trigger counts inside one block.

    Expected:
        240 normal face trials
        24 oddball trials
        each identity 1-60 occurs four times
    """

    events = events_from_annotations_keep_codes(
        block
    )

    normal_events = events[
        np.isin(
            events[:, 2],
            STIMULUS_CODES,
        )
    ]

    oddball_events = events[
        np.isin(
            events[:, 2],
            ODDBALL_CODES,
        )
    ]

    identity_counts = (
        pd.Series(
            normal_events[:, 2],
            name="identity",
        )
        .value_counts()
        .reindex(
            STIMULUS_CODES,
            fill_value=0,
        )
        .sort_index()
        .rename("count")
        .reset_index()
    )

    identity_counts["block_code"] = (
        block_code
    )

    print(
        f"Block {block_code}: "
        f"{len(normal_events)} normal trials, "
        f"{len(oddball_events)} oddball trials."
    )

    incorrect_repetitions = (
        identity_counts[
            identity_counts["count"]
            != EXPECTED_REPETITIONS_PER_IDENTITY
        ]
    )
    
    oddball_count_table = (
        pd.Series(
            oddball_events[:, 2],
            name="oddball_code",
        )
        .value_counts()
        .sort_index()
    )

    print(
        f"\nBlock {block_code} oddball counts:"
    )

    print(
        oddball_count_table.to_string()
    )

    if (
        len(oddball_events)
        != EXPECTED_ODDBALL_TRIALS_PER_BLOCK
    ):

        print(
            f"\nDetailed oddball events for "
            f"block {block_code}:"
        )

        for oddball_event in oddball_events:

            event_sample = int(
                oddball_event[0]
            )

            event_code = int(
                oddball_event[2]
            )

            matching_index = np.flatnonzero(
                events[:, 0] == event_sample
            )

            if len(matching_index) != 1:
                continue

            event_index = int(
                matching_index[0]
            )

            window_start = max(
                0,
                event_index - 2,
            )

            window_stop = min(
                len(events),
                event_index + 3,
            )

            nearby_events = events[
                window_start:window_stop
            ]

            nearby_description = [
                {
                    "sample": int(row[0]),
                    "code": int(row[2]),
                    "relative_ms": float(
                        (
                            int(row[0])
                            - event_sample
                        )
                        / block.info["sfreq"]
                        * 1000
                    ),
                }
                for row in nearby_events
            ]

            print(
                f"  Oddball code {event_code} "
                f"at sample {event_sample}"
            )

            print(
                f"    Nearby events: "
                f"{nearby_description}"
            )

    validation_errors = []

    if (
        len(normal_events)
        != EXPECTED_NORMAL_TRIALS_PER_BLOCK
    ):

        validation_errors.append(
            f"expected "
            f"{EXPECTED_NORMAL_TRIALS_PER_BLOCK} "
            f"normal trials, found "
            f"{len(normal_events)}"
        )

    if (
        len(oddball_events)
        != EXPECTED_ODDBALL_TRIALS_PER_BLOCK
    ):

        validation_errors.append(
            f"expected "
            f"{EXPECTED_ODDBALL_TRIALS_PER_BLOCK} "
            f"oddball trials, found "
            f"{len(oddball_events)}"
        )

    if not incorrect_repetitions.empty:

        bad_counts = (
            incorrect_repetitions[
                ["identity", "count"]
            ]
            .to_dict(
                orient="records"
            )
        )

        validation_errors.append(
            f"incorrect identity repetitions: "
            f"{bad_counts}"
        )

    if validation_errors:
        print(
            f"[WARNING] Block {block_code}: "
            + "; ".join(validation_errors)
        )

    print(
        f"Block {block_code}: event validation passed."
    )

    return identity_counts

# =============================================================================
# STEP 12 — SEGMENT NORMAL STIMULUS TRIALS 1-60
#
# Letswave:
# Preprocess → Segmentation → Segmentation relative to events
#
# Select stimulus codes: 1-60
# Epoch start: -0.1 seconds
# Epoch duration: 1 second
# Prefix: trl
#
# This step is performed separately for every block.
# =============================================================================

TRIAL_START_SECONDS = -0.1

TRIAL_DURATION_SECONDS = 1.0


def create_trial_metadata(
    trial_events: np.ndarray,
    block: mne.io.BaseRaw,
    block_code: int,
) -> pd.DataFrame:
    """
    Create one metadata row for every normal face trial.

    Metadata includes:
        block
        identity
        race
        repetition number within block
        block-relative event time
    """

    sfreq = float(block.info["sfreq"])

    repetition_counts = {
        identity: 0
        for identity in STIMULUS_CODES
    }

    rows = []

    for trial_number, event in enumerate(
        trial_events,
        start=1,
    ):

        identity = int(event[2])

        repetition_counts[identity] += 1

        if identity in ASIAN_STIMULUS_CODES:

            race = "Asian"

        else:

            race = "Caucasian"

        # event[0] and block.first_samp are both expressed in the same
        # absolute sample coordinate.
        block_relative_sample = (
            int(event[0])
            - int(block.first_samp)
        )

        rows.append(
            {
                "block_code": block_code,
                "trial_within_block": trial_number,
                "identity": identity,
                "race": race,
                "identity_repetition_within_block":
                    repetition_counts[identity],
                "event_sample_absolute": int(event[0]),
                "event_sample_within_block":
                    block_relative_sample,
                "event_time_seconds_within_block":
                    block_relative_sample / sfreq,
            }
        )

    return pd.DataFrame(rows)


def segment_stimulus_trials(
    block: mne.io.BaseRaw,
    block_code: int,
) -> tuple[mne.Epochs, pd.DataFrame]:
    """
    Segment stimulus codes 1-60 from one block.

    The returned object contains only normal face trials.
    Oddball triggers 101-160 are not included.

    Epoch:
        start = -0.1 seconds
        duration = 1 second
    """

    all_events = events_from_annotations_keep_codes(
        block
    )

    trial_events = all_events[
        np.isin(
            all_events[:, 2],
            STIMULUS_CODES,
        )
    ].copy()

    if len(trial_events) == 0:

        raise RuntimeError(
            f"Block {block_code}: no stimulus events 1-60 found."
        )

    sfreq = float(block.info["sfreq"])

    # MNE includes both tmin and tmax.
    # Subtract one sample to obtain exactly duration × sfreq samples.
    trial_samples = int(
        round(
            TRIAL_DURATION_SECONDS
            * sfreq
        )
    )

    trial_tmax = (
        TRIAL_START_SECONDS
        + (
            trial_samples - 1
        ) / sfreq
    )

    metadata = create_trial_metadata(
        trial_events=trial_events,
        block=block,
        block_code=block_code,
    )

    event_id = {
        f"id_{identity:02d}": identity
        for identity in STIMULUS_CODES
    }

    epochs = mne.Epochs(
        raw=block,
        events=trial_events,
        event_id=event_id,
        tmin=TRIAL_START_SECONDS,
        tmax=trial_tmax,
        baseline=None,
        detrend=None,
        picks="eeg",
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=True,
        event_repeated="error",
        metadata=metadata,
        verbose="INFO",
    )
    
    dropped_epoch_indices = [
        index
        for index, reasons in enumerate(
            epochs.drop_log
        )
        if reasons
    ]

    dropped_epoch_rows = []

    raw_last_sample = (
        int(block.first_samp)
        + int(block.n_times)
        - 1
    )

    if dropped_epoch_indices:

        print("\nDropped epoch(s):")

        for candidate_index in dropped_epoch_indices:

            dropped_event = trial_events[
                candidate_index
            ]

            event_sample = int(
                dropped_event[0]
            )

            identity = int(
                dropped_event[2]
            )

            reasons = [
                str(reason)
                for reason in epochs.drop_log[
                    candidate_index
                ]
            ]

            reason_text = ", ".join(
                reasons
            )

            required_epoch_end = (
                event_sample
                + int(
                    round(
                        trial_tmax
                        * sfreq
                    )
                )
            )

            missing_samples = max(
                0,
                required_epoch_end
                - raw_last_sample,
            )

            missing_seconds = (
                missing_samples / sfreq
            )

            race = (
                "Asian"
                if identity
                in ASIAN_STIMULUS_CODES
                else "Caucasian"
            )

            dropped_epoch_rows.append(
                {
                    "subject": SUBJECT,
                    "session": SESSION,
                    "block_code": block_code,
                    "identity": identity,
                    "race": race,
                    "trial_within_block": int(
                        metadata.iloc[candidate_index][
                            "trial_within_block"
                        ]
                    ),
                    "identity_repetition_within_block": int(
                        metadata.iloc[candidate_index][
                            "identity_repetition_within_block"
                        ]
                    ),
                    "event_sample_absolute": event_sample,
                    "mne_reason": reason_text,
                    "missing_samples": missing_samples,
                    "missing_seconds": missing_seconds,
                    "epochs_retained": len(epochs),
                    "epochs_expected": len(trial_events),
                }
            )

            print(
                f"  Block: {block_code}"
            )

            print(
                f"  Identity: {identity}"
            )

            print(
                f"  Reason: {reason_text}"
            )

            print(
                f"  Missing samples: "
                f"{missing_samples} "
                f"({missing_seconds:.3f} s)"
            )

        print(
            f"  Remaining trials in block: "
            f"{len(epochs)} / "
            f"{len(trial_events)}"
        )

    dropped_epoch_report = pd.DataFrame(
        dropped_epoch_rows,
        columns=[
            "subject",
            "session",
            "block_code",
            "identity",
            "race",
            "trial_within_block",
            "identity_repetition_within_block",
            "event_sample_absolute",
            "mne_reason",
            "missing_samples",
            "missing_seconds",
            "epochs_retained",
            "epochs_expected",
        ],
    )

    if epochs.metadata is None:
        raise RuntimeError(
            f"Block {block_code}: metadata is missing after epoch creation."
        )

    if len(epochs) != len(epochs.metadata):
        raise RuntimeError(
            f"Block {block_code}: epoch count "
            f"({len(epochs)}) does not match metadata rows "
            f"({len(epochs.metadata)})."
        )

    dropped_by_boundary = [
        index
        for index, reasons in enumerate(
            epochs.drop_log
        )
        if any(
            str(reason).lower().startswith(
                "bad boundary"
            )
            for reason in reasons
        )
    ]

    if dropped_by_boundary:
        print(
            f"[WARNING] Block {block_code}: "
            f"{len(dropped_by_boundary)} trial(s) were dropped "
            "because they overlapped a BDF concatenation boundary. "
            f"Candidate indices: {dropped_by_boundary}"
        )

    if epochs.get_data(copy=False).shape[-1] != trial_samples:

        raise RuntimeError(
            f"Block {block_code}: expected "
            f"{trial_samples} samples per trial, "
            f"but obtained "
            f"{epochs.get_data(copy=False).shape[-1]}."
        )

    print(
        f"trl block {block_code}: "
        f"{len(epochs)} trials, "
        f"{len(epochs.times)} samples per trial, "
        f"{epochs.times[0]:.6f} to "
        f"{epochs.times[-1]:.6f} seconds."
    )

    return epochs, dropped_epoch_report

# =============================================================================
# STEP 13 — DC REMOVAL WITHOUT LINEAR DETREND
#
# Letswave:
# Preprocess → DC Removal
#
# Do not select linear detrend.
# Prefix: dc
#
# Performed separately for every block.
# =============================================================================

def remove_dc_offset(
    epochs: mne.Epochs,
) -> mne.Epochs:
    """
    Remove the constant/DC offset from every epoch and EEG channel.

    This does NOT perform linear detrending.
    """

    output = epochs.copy()

    epoch_channel_means = output._data.mean(
        axis=-1,
        keepdims=True,
    )

    output._data -= epoch_channel_means

    return output

# =============================================================================
# STEP 14 — AMPLITUDE-BASED ARTIFACT REJECTION
#
# Letswave:
# Preprocess → Artifacts → Reject epochs
# Select amplitude criterion
# Threshold: 700 microvolts
# Select all electrodes
# Prefix: ar
#
# Criterion implemented here:
# Reject an epoch if ANY EEG channel contains ANY sample:
#
#     amplitude > +700 µV
#                  OR
#     amplitude < -700 µV
#
# This is performed separately for every block.
# =============================================================================

ARTIFACT_THRESHOLD_UV = 700.0

ARTIFACT_THRESHOLD_V = (
    ARTIFACT_THRESHOLD_UV
    * 1e-6
)


def reject_epochs_by_absolute_amplitude(
    epochs: mne.Epochs,
    block_code: int,
) -> tuple[mne.Epochs, pd.DataFrame]:
    """
    Reject epochs exceeding ±700 µV on any EEG electrode.

    Parameters
    ----------
    epochs
        DC-removed epochs from one block.

    block_code
        Testing block trigger code, 202-217.

    Returns
    -------
    clean_epochs
        Epochs after artifact rejection.

    rejection_report
        One row for every rejected epoch.
    """

    data = epochs.get_data(
        picks="eeg",
        copy=False,
    )

    # Shape:
    # epochs × channels × time
    absolute_data = np.abs(data)

    # Maximum absolute amplitude for every epoch/channel.
    max_absolute_by_channel = absolute_data.max(
        axis=-1
    )

    # Reject an epoch when any channel exceeds 700 µV.
    bad_epoch_mask = np.any(
        max_absolute_by_channel
        > ARTIFACT_THRESHOLD_V,
        axis=1,
    )

    bad_epoch_indices = np.flatnonzero(
        bad_epoch_mask
    )

    report_rows = []

    for epoch_index in bad_epoch_indices:

        violating_channel_indices = np.flatnonzero(
            max_absolute_by_channel[epoch_index]
            > ARTIFACT_THRESHOLD_V
        )

        violating_channel_names = [
            epochs.ch_names[channel_index]
            for channel_index
            in violating_channel_indices
        ]

        metadata = {}

        if epochs.metadata is not None:

            metadata = (
                epochs.metadata
                .iloc[epoch_index]
                .to_dict()
            )

        report_rows.append(
            {
                **metadata,
                "block_code": block_code,
                "epoch_index_before_rejection":
                    int(epoch_index),
                "maximum_absolute_amplitude_uV":
                    float(
                        max_absolute_by_channel[
                            epoch_index
                        ].max()
                        * 1e6
                    ),
                "violating_channels":
                    ",".join(
                        violating_channel_names
                    ),
                "criterion":
                    "absolute amplitude > ±700 µV",
            }
        )

    clean_epochs = epochs.copy()

    if len(bad_epoch_indices) > 0:

        clean_epochs.drop(
            bad_epoch_indices.tolist(),
            reason="ABS_AMPLITUDE_GT_700uV",
            verbose="INFO",
        )

    rejection_report = pd.DataFrame(
        report_rows
    )

    print(
        f"ar block {block_code}: "
        f"rejected {len(bad_epoch_indices)} / "
        f"{len(epochs)} epochs; "
        f"retained {len(clean_epochs)}."
    )

    return clean_epochs, rejection_report

# =============================================================================
# RUN STEP 1-10
# IMPORT → COORDINATES → EVENT CLEANING → CH → BUT → BLK
# =============================================================================

def bdf_part_number(
    path: Path,
    subject: str,
    session: str,
) -> int:
    """
    Return the acquisition part number from a BDF filename.

    Examples
    --------
    subj17_S1.bdf   -> 1
    subj17_S1_2.bdf -> 2
    subj17_S1_3.bdf -> 3
    """

    base_name = (
        f"{subject}_{session}"
    )

    match = re.fullmatch(
        rf"{re.escape(base_name)}"
        rf"(?:_(\d+))?"
        rf"\.bdf",
        path.name,
        re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Unexpected BDF filename: {path.name}"
        )

    suffix_number = match.group(1)

    if suffix_number is None:
        return 1

    return int(suffix_number)

bdf_matches = [
    path
    for path in BDF_DIRECTORY.iterdir()
    if path.is_file()
    and re.fullmatch(
        rf"{re.escape(SUBJECT)}_"
        rf"{re.escape(SESSION)}"
        rf"(?:_\d+)?\.bdf",
        path.name,
        re.IGNORECASE,
    )
]

bdf_matches = sorted(
    bdf_matches,
    key=lambda path: bdf_part_number(
        path=path,
        subject=SUBJECT,
        session=SESSION,
    ),
)

if len(bdf_matches) == 0:
    raise FileNotFoundError(
        f"No BDF files were found for "
        f"{SUBJECT} {SESSION} in "
        f"{BDF_DIRECTORY}."
    )

part_numbers = [
    bdf_part_number(
        path=path,
        subject=SUBJECT,
        session=SESSION,
    )
    for path in bdf_matches
]

if len(part_numbers) != len(set(part_numbers)):
    raise RuntimeError(
        "Duplicate BDF part numbers were found. "
        f"Files: {[path.name for path in bdf_matches]}, "
        f"part numbers: {part_numbers}"
    )

expected_part_numbers = list(
    range(
        1,
        len(bdf_matches) + 1,
    )
)

if part_numbers != expected_part_numbers:
    raise RuntimeError(
        "The BDF file sequence is incomplete or "
        "numbered unexpectedly. "
        f"Found parts {part_numbers}; expected "
        f"{expected_part_numbers}. "
        f"Files: {[path.name for path in bdf_matches]}"
    )

print(
    "Using BDF file(s) in this order:"
)

for bdf_file in bdf_matches:
    print(
        f"  {bdf_file.resolve()}"
    )
    
output_root = (
    Path("preprocessed")
    / SUBJECT
    / SESSION
)

import_directory = (
    output_root
    / "00_import"
)

ch_directory = (
    output_root
    / "01_ch"
)

but_directory = (
    output_root
    / "02_but"
)

blk_directory = (
    output_root
    / "03_blk"
)

audit_directory = (
    output_root
    / "audit"
)

for directory in (
    import_directory,
    ch_directory,
    but_directory,
    blk_directory,
    audit_directory,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# -----------------------------------------------------------------------------
# STEP 1 — Import BDF
# -----------------------------------------------------------------------------

raw = import_bdf_files(
    bdf_files=bdf_matches,
    status_channel="Status",
)

if "Status" not in raw.ch_names:
    raise RuntimeError(
        "Status channel was not found. "
        f"Available channels: {raw.ch_names}"
    )

if not np.isclose(
    raw.info["sfreq"],
    EXPECTED_SFREQ,
):
    raise RuntimeError(
        f"Expected {EXPECTED_SFREQ} Hz, "
        f"but BDF sampling rate is "
        f"{raw.info['sfreq']} Hz."
    )

raw.save(
    import_directory
    / f"import_{SUBJECT}_{SESSION}_raw.fif",
    overwrite=True,
)

# -----------------------------------------------------------------------------
# Extract events before deleting Status
# -----------------------------------------------------------------------------

all_masked_events = extract_events(
    raw=raw,
    status_channel="Status",
)

# Do not remove events solely because they are close to a BDF
# concatenation boundary. Valid block triggers may occur immediately
# after the next BDF file begins.
original_events = all_masked_events.copy()

boundary_artifact_events = np.empty(
    (0, 3),
    dtype=int,
)

print(
    "\nBoundary-adjacent trigger removal is disabled. "
    f"Retained all {len(original_events)} masked events."
)

np.savetxt(
    audit_directory
    / "events_boundary_artifacts.tsv",
    boundary_artifact_events,
    fmt="%d",
    delimiter="\t",
    header="sample\tprevious\tcode",
    comments="",
)


event_codes = np.unique(
    original_events[:, 2]
)

bad_codes = event_codes[
    (event_codes < 0)
    | (event_codes > 255)
]

if len(bad_codes) > 0:
    raise RuntimeError(
        "Trigger masking failed. "
        f"Found event codes outside 0-255: {bad_codes.tolist()}"
    )

np.savetxt(
    audit_directory
    / "events_original.tsv",
    original_events,
    fmt="%d",
    delimiter="\t",
    header="sample\tprevious\tcode",
    comments="",
)


# -----------------------------------------------------------------------------
# STEP 2 — Assign coordinates
# -----------------------------------------------------------------------------

coordinate_raw = assign_electrode_coordinates(
    raw
)


# -----------------------------------------------------------------------------
# STEP 3-4 — Remove unwanted events, then remove EX and Status
# -----------------------------------------------------------------------------

retained_events, removed_events = (
    remove_unwanted_events(
        original_events
    )
)

np.savetxt(
    audit_directory
    / "events_retained.tsv",
    retained_events,
    fmt="%d",
    delimiter="\t",
    header="sample\tprevious\tcode",
    comments="",
)

np.savetxt(
    audit_directory
    / "events_removed.tsv",
    removed_events,
    fmt="%d",
    delimiter="\t",
    header="sample\tprevious\tcode",
    comments="",
)

annotation_raw = events_to_annotations(
    raw=coordinate_raw,
    retained_events=retained_events,
)

annotation_events = events_from_annotations_keep_codes(
    annotation_raw
)

if annotation_events.shape != retained_events.shape:
    raise RuntimeError(
        "Annotation event array shape does not match retained events: "
        f"{annotation_events.shape} vs {retained_events.shape}."
    )

retained_sample_and_code = retained_events[:, [0, 2]]
annotation_sample_and_code = annotation_events[:, [0, 2]]

difference = (
    annotation_sample_and_code
    - retained_sample_and_code
)

mismatch_rows = np.flatnonzero(
    np.any(
        difference != 0,
        axis=1,
    )
)

if len(mismatch_rows) > 0:

    print("\nFirst annotation conversion mismatches:")

    for row in mismatch_rows[:20]:

        print(
            f"row {row}: "
            f"retained_sample_code="
            f"{retained_sample_and_code[row].tolist()} "
            f"annotation_sample_code="
            f"{annotation_sample_and_code[row].tolist()} "
            f"difference={difference[row].tolist()}"
        )

    raise RuntimeError(
        "Event sample positions or codes changed during "
        "annotation conversion. "
        f"{len(mismatch_rows)} mismatching rows."
    )

print(
    "Annotation round-trip validation passed: "
    "event samples and codes were preserved."
)
    
ch_raw = remove_channels(
    raw=annotation_raw,
    status_channel="Status",
)

ch_raw.save(
    ch_directory
    / f"ch_{SUBJECT}_{SESSION}_raw.fif",
    overwrite=True,
)


# -----------------------------------------------------------------------------
# STEP 5 — Butterworth filter
# -----------------------------------------------------------------------------

but_raw = butterworth_filter(
    raw=ch_raw,
    n_jobs=1,
)

but_raw.save(
    but_directory
    / f"but_{SUBJECT}_{SESSION}_raw.fif",
    overwrite=True,
)


# -----------------------------------------------------------------------------
# STEP 6 — Find and segment blocks 202-217
# -----------------------------------------------------------------------------

block_events = find_block_events(
    but_raw
)

blocks = segment_blocks(
    raw=but_raw,
    block_events=block_events,
)

for block_code in TEST_BLOCKS:

    block_path = (
        blk_directory
        / (
            f"blk{block_code}_"
            f"{SUBJECT}_{SESSION}_raw.fif"
        )
    )

    blocks[block_code].save(
        block_path,
        overwrite=True,
    )

print(
    f"Created {len(blocks)} testing blocks."
)

# =============================================================================
# RUN STEP 11-14 FOR EVERY BLOCK INDIVIDUALLY
# =============================================================================

trl_directory = (
    output_root
    / "04_trl"
)

dc_directory = (
    output_root
    / "05_dc"
)

ar_directory = (
    output_root
    / "06_ar"
)

audit_directory = (
    output_root
    / "audit"
)

for directory in (
    trl_directory,
    dc_directory,
    ar_directory,
    audit_directory,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


trl_blocks = {}

dc_blocks = {}

ar_blocks = {}

all_identity_count_tables = []

all_rejection_reports = []

all_dropped_epoch_reports = []


for block_code in TEST_BLOCKS:

    print("\n" + "=" * 70)
    print(f"PROCESSING BLOCK {block_code}")
    print("=" * 70)

    block = blocks[block_code]

    # -------------------------------------------------------------------------
    # Validate experimental event structure.
    # -------------------------------------------------------------------------

    identity_counts = validate_block_events(
        block=block,
        block_code=block_code,
    )

    all_identity_count_tables.append(
        identity_counts
    )

    # -------------------------------------------------------------------------
    # Prefix trl
    # -------------------------------------------------------------------------

    (
        trl_epochs,
        dropped_epoch_report,
    ) = segment_stimulus_trials(
        block=block,
        block_code=block_code,
    )

    # Only append a report when this block contains dropped epochs.
    if not dropped_epoch_report.empty:

        all_dropped_epoch_reports.append(
            dropped_epoch_report
        )

    # These steps must run for every block, regardless of whether an epoch
    # was dropped.
    trl_blocks[block_code] = trl_epochs

    trl_path = (
        trl_directory
        / f"trl_blk{block_code}-epo.fif"
    )

    trl_epochs.save(
        trl_path,
        overwrite=True,
    )

    # -------------------------------------------------------------------------
    # Prefix dc
    # -------------------------------------------------------------------------

    dc_epochs = remove_dc_offset(
        trl_epochs
    )

    dc_blocks[block_code] = dc_epochs

    dc_path = (
        dc_directory
        / f"dc_blk{block_code}-epo.fif"
    )

    dc_epochs.save(
        dc_path,
        overwrite=True,
    )

    # -------------------------------------------------------------------------
    # Prefix ar
    # -------------------------------------------------------------------------

    (
        ar_epochs,
        rejection_report,
    ) = reject_epochs_by_absolute_amplitude(
        epochs=dc_epochs,
        block_code=block_code,
    )

    ar_blocks[block_code] = ar_epochs

    ar_path = (
        ar_directory
        / f"ar_blk{block_code}-epo.fif"
    )

    ar_epochs.save(
        ar_path,
        overwrite=True,
    )

    if not rejection_report.empty:

        all_rejection_reports.append(
            rejection_report
        )


# =============================================================================
# SAVE AUDIT TABLES
# =============================================================================

identity_count_report = pd.concat(
    all_identity_count_tables,
    ignore_index=True,
)

identity_count_report.to_csv(
    audit_directory
    / "identity_counts_by_block.csv",
    index=False,
)

if all_dropped_epoch_reports:

    full_dropped_epoch_report = pd.concat(
        all_dropped_epoch_reports,
        ignore_index=True,
    )

else:

    full_dropped_epoch_report = pd.DataFrame(
        columns=[
            "subject",
            "session",
            "block_code",
            "identity",
            "race",
            "trial_within_block",
            "identity_repetition_within_block",
            "event_sample_absolute",
            "mne_reason",
            "missing_samples",
            "missing_seconds",
            "epochs_retained",
            "epochs_expected",
        ]
    )


full_dropped_epoch_report.to_csv(
    audit_directory
    / "dropped_epochs.csv",
    index=False,
)

if all_rejection_reports:

    full_rejection_report = pd.concat(
        all_rejection_reports,
        ignore_index=True,
    )

else:

    full_rejection_report = pd.DataFrame(
        columns=[
            "block_code",
            "epoch_index_before_rejection",
            "maximum_absolute_amplitude_uV",
            "violating_channels",
            "criterion",
        ]
    )


full_rejection_report.to_csv(
    ar_directory
    / "ar_rejection_report.csv",
    index=False,
)


summary_rows = []

for block_code in TEST_BLOCKS:

    summary_rows.append(
        {
            "block_code": block_code,
            "trl_epochs_before_rejection":
                len(trl_blocks[block_code]),
            "ar_epochs_after_rejection":
                len(ar_blocks[block_code]),
            "epochs_rejected":
                len(trl_blocks[block_code])
                - len(ar_blocks[block_code]),
        }
    )


pd.DataFrame(
    summary_rows
).to_csv(
    audit_directory
    / "artifact_rejection_summary.csv",
    index=False,
)


print("\nCompleted prefixes:")

print("trl")

print("dc")

print("ar")

# =============================================================================
# STEP 15 — COMPUTE ICA MATRIX
#
# Letswave:
# Preprocess
# → Spatial filters
# → Compute ICA Matrix (merge datasets)
#
# Select all artifact-cleaned blocks within one session.
#
# Prefix: ica
#
# MNE implementation:
# 1. Merge all ar blocks from one session.
# 2. Fit one ICA model on the merged Epochs.
# 3. Do not overwrite the original ar files.
#
# The ar data have been high-pass filtered at 0.1 Hz and have not yet received
# baseline correction, matching the supplied protocol order.
# =============================================================================

ICA_RANDOM_STATE = 97

ICA_METHOD = "fastica"


def merge_ar_blocks_for_ica(
    ar_blocks: dict[int, mne.Epochs],
) -> mne.Epochs:
    """
    Merge all artifact-cleaned blocks from one session for ICA fitting.

    The individual ar block objects are copied and therefore are not
    overwritten or modified.
    """

    missing_blocks = [
        block_code
        for block_code in TEST_BLOCKS
        if block_code not in ar_blocks
    ]

    if missing_blocks:

        raise RuntimeError(
            f"Missing ar blocks for ICA: {missing_blocks}"
        )

    merged_epochs = mne.concatenate_epochs(
        [
            ar_blocks[block_code].copy()
            for block_code in TEST_BLOCKS
        ],
        add_offset=True,
        on_mismatch="raise",
        verbose="INFO",
    )

    print(
        f"ICA input: {len(merged_epochs)} epochs "
        f"from {len(TEST_BLOCKS)} blocks."
    )

    return merged_epochs


def compute_session_ica(
    merged_epochs: mne.Epochs,
) -> ICA:
    """
    Compute one ICA matrix for all blocks in one session.

    n_components=None tells MNE to retain the available data rank rather than
    manually selecting a smaller component count.
    """

    ica = ICA(
        n_components=None,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )

    ica.fit(
        merged_epochs,
        picks="eeg",
        reject_by_annotation=True,
        verbose="INFO",
    )

    print(
        f"ICA fitted with {ica.n_components_} components."
    )

    print(
        f"ICA iterations: {ica.n_iter_}"
    )

    return ica

# =============================================================================
# STEP 16 — SAVE ICA MATRIX AND COMPONENT FIGURES
#
# Prefix: ica
#
# Outputs:
# ica_merged_ar_subjXX_SX-epo.fif
# ica_subjXX_SX-ica.fif
# ICA component topography figures
# =============================================================================
def save_ica_outputs(
    merged_epochs: mne.Epochs,
    ica: ICA,
    output_directory: Path,
    subject: str,
    session: str,
    components_to_remove: list[int],
    overwrite: bool = True,
) -> ICA:
    """
    Save:

    1. Merged epochs used to fit ICA.
    2. Original fitted ICA with no excluded components.
    3. ICA component topographies.
    4. Detailed diagnostic plots for selected components.
    5. Overlay comparing EEG before and after ICA exclusion.
    6. Final ICA file containing ica.exclude.

    MNE component numbering is zero-based:
        ICA000 -> index 0
        ICA001 -> index 1
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    invalid_components = [
        component
        for component in components_to_remove
        if component < 0
        or component >= ica.n_components_
    ]

    if invalid_components:

        raise IndexError(
            f"Invalid ICA component indices: "
            f"{invalid_components}. "
            f"Available indices are "
            f"0-{ica.n_components_ - 1}."
        )

    # -------------------------------------------------------------------------
    # Save the merged epochs used for ICA fitting.
    # -------------------------------------------------------------------------

    merged_path = (
        output_directory
        / (
            f"ica_merged_ar_"
            f"{subject}_{session}-epo.fif"
        )
    )

    merged_epochs.save(
        merged_path,
        overwrite=overwrite,
    )

    # -------------------------------------------------------------------------
    # Save the original ICA before excluding any component.
    # -------------------------------------------------------------------------

    ica.exclude = []

    unreviewed_ica_path = (
        output_directory
        / (
            f"ica_unreviewed_"
            f"{subject}_{session}-ica.fif"
        )
    )

    ica.save(
        unreviewed_ica_path,
        overwrite=overwrite,
    )

    print(
        f"Saved unreviewed ICA: "
        f"{unreviewed_ica_path}"
    )

    # -------------------------------------------------------------------------
    # Save all ICA component topographies.
    # -------------------------------------------------------------------------

    try:

        component_figures = ica.plot_components(
            show=False,
        )

        if not isinstance(
            component_figures,
            list,
        ):

            component_figures = [
                component_figures
            ]

        for page_number, figure in enumerate(
            component_figures,
            start=1,
        ):

            figure.savefig(
                figure_directory
                / (
                    f"ica_components_"
                    f"{subject}_{session}_"
                    f"page-{page_number:02d}.png"
                ),
                dpi=160,
                bbox_inches="tight",
            )

            plt.close(
                figure
            )

    except Exception as error:

        print(
            "[WARNING] ICA component maps "
            "could not be saved:"
        )

        print(error)

    # -------------------------------------------------------------------------
    # Mark selected components for exclusion.
    # -------------------------------------------------------------------------

    ica.exclude = list(
        components_to_remove
    )

    print(
        "ICA components selected for removal: "
        f"{ica.exclude}"
    )

    # -------------------------------------------------------------------------
    # Save detailed properties for the selected components.
    # -------------------------------------------------------------------------

    if ica.exclude:

        try:

            property_figures = (
                ica.plot_properties(
                    merged_epochs,
                    picks=ica.exclude,
                    show=False,
                )
            )

            if not isinstance(
                property_figures,
                list,
            ):

                property_figures = [
                    property_figures
                ]

            for component_index, figure in zip(
                ica.exclude,
                property_figures,
            ):

                figure.savefig(
                    figure_directory
                    / (
                        f"ica_properties_"
                        f"ICA{component_index:03d}_"
                        f"{subject}_{session}.png"
                    ),
                    dpi=160,
                    bbox_inches="tight",
                )

                plt.close(
                    figure
                )
                
        except Exception as error:

            print(
                "[WARNING] ICA property figures "
                "could not be saved:"
            )

            print(error)

    # -------------------------------------------------------------------------
    # Save EEG overlay before and after ICA exclusion.
    # -------------------------------------------------------------------------

    if ica.exclude:

        try:

            overlay_figure = ica.plot_overlay(
                merged_epochs,
                exclude=ica.exclude,
                picks="eeg",
                show=False,
            )

            overlay_figure.savefig(
                figure_directory
                / (
                    f"ica_overlay_exclude-"
                    f"{'-'.join(map(str, ica.exclude))}_"
                    f"{subject}_{session}.png"
                ),
                dpi=160,
                bbox_inches="tight",
            )

            plt.close(
                overlay_figure
            )

        except Exception as error:

            print(
                "[WARNING] ICA overlay figure "
                "could not be saved:"
            )

            print(error)

    # -------------------------------------------------------------------------
    # Save ICA again with exclusion information stored in ica.exclude.
    # -------------------------------------------------------------------------

    marked_ica_path = (
        output_directory
        / (
            f"ica_marked_"
            f"{subject}_{session}-ica.fif"
        )
    )

    ica.save(
        marked_ica_path,
        overwrite=overwrite,
    )

    print(
        f"Saved marked ICA: "
        f"{marked_ica_path}"
    )

    return ica

# =============================================================================
# STEP 17 — SELECT ARTIFACT ICA COMPONENTS
# Components selected for removal.
#
# Component numbering in MNE is zero-based:
#
# ICA000 -> index 0
# ICA001 -> index 1
#
# The original Letswave workflow removed the first visually identified
# blink component. This pipeline currently removes the first ICA component
# (index 0) to reproduce that workflow.
#
# Users should verify component topographies before changing this list.
# =============================================================================

ICA_COMPONENTS_TO_REMOVE = [
    0,
]

# =============================================================================
# STEP 18 — APPLY ICA SPATIAL FILTER
#
# Letswave:
# Preprocess
# → Spatial filters
# → ICA apply spatial filter
#
# Select all blocks.
# Remove selected eye-blink ICs.
#
# Prefix: icfilt
#
# The same session-level ICA matrix is applied independently to every ar block.
# =============================================================================


def apply_ica_to_block(
    epochs: mne.Epochs,
    ica: ICA,
) -> mne.Epochs:
    """
    Apply the fitted session ICA to one artifact-cleaned block.

    The input ar block is copied and is not overwritten.
    """

    output = epochs.copy()

    ica.apply(
        output,
        exclude=ica.exclude,
        verbose="INFO",
    )

    return output

# =============================================================================
# STEP 19 — DC REMOVAL AND LINEAR DETREND
#
# Letswave:
# Preprocess
# → DC Removal and linear detrend
#
# Check:
# Apply linear detrending in addition to DC removal
#
# Prefix: dt
#
# scipy.signal.detrend(type="linear") removes the fitted linear trend,
# including its constant/DC component, separately for every
# epoch × channel.
# =============================================================================


def remove_dc_and_linear_trend(
    epochs: mne.Epochs,
) -> mne.Epochs:
    """
    Remove the linear trend and constant offset from every epoch/channel.
    """

    output = epochs.copy()

    output._data[:] = scipy_detrend(
        output._data,
        axis=-1,
        type="linear",
        overwrite_data=False,
    )

    return output

# =============================================================================
# STEP 20 — AVERAGE REREFERENCE
#
# Letswave:
# Preprocess
# → Rereference
#
# Select all EEG electrodes on both sides.
#
# Prefix: reref
#
# MNE implementation:
# Use the mean of all remaining EEG electrodes as the reference.
# =============================================================================


def average_rereference(
    epochs: mne.Epochs,
) -> mne.Epochs:
    """
    Rereference one block to the average of all EEG electrodes.
    """

    output = epochs.copy()

    output.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="INFO",
    )

    return output

# =============================================================================
# STEP 21 — BASELINE CORRECTION
#
# Letswave:
# Preprocess
# → Baseline operations
# → Baseline correction
#
# Baseline interval:
# -0.1 to 0 seconds
#
# Prefix: bl
#
# Performed separately for every block.
# =============================================================================

BASELINE_INTERVAL = (
    -0.1,
    0.0,
)


def apply_baseline_correction(
    epochs: mne.Epochs,
) -> mne.Epochs:
    """
    Subtract the mean amplitude from -0.1 to 0 seconds separately for every
    epoch and EEG channel.
    """

    output = epochs.copy()

    output.apply_baseline(
        baseline=BASELINE_INTERVAL,
        verbose="INFO",
    )

    return output

# =============================================================================
# RUN ICA AND POST-ICA PROCESSING
# =============================================================================


ica_directory = (
    output_root
    / "07_ica"
)

icfilt_directory = (
    output_root
    / "08_icfilt"
)

dt_directory = (
    output_root
    / "09_dt"
)

reref_directory = (
    output_root
    / "10_reref"
)

bl_directory = (
    output_root
    / "11_bl"
)


for directory in (
    ica_directory,
    icfilt_directory,
    dt_directory,
    reref_directory,
    bl_directory,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# -----------------------------------------------------------------------------
# Prefix ica
# -----------------------------------------------------------------------------

merged_ar_epochs = merge_ar_blocks_for_ica(
    ar_blocks
)

session_ica = compute_session_ica(
    merged_ar_epochs
)

session_ica = save_ica_outputs(
    merged_epochs=merged_ar_epochs,
    ica=session_ica,
    output_directory=ica_directory,
    subject=SUBJECT,
    session=SESSION,
    components_to_remove=(
        ICA_COMPONENTS_TO_REMOVE
    ),
    overwrite=True,
)


# -----------------------------------------------------------------------------
# Prefixes icfilt, dt, reref, and bl
# Each step is performed separately for every block.
# -----------------------------------------------------------------------------

icfilt_blocks = {}

dt_blocks = {}

reref_blocks = {}

bl_blocks = {}


for block_code in TEST_BLOCKS:

    print("\n" + "=" * 70)

    print(
        f"POST-ICA PROCESSING BLOCK {block_code}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Prefix icfilt
    # -------------------------------------------------------------------------

    icfilt_epochs = apply_ica_to_block(
        epochs=ar_blocks[block_code],
        ica=session_ica,
    )

    icfilt_blocks[block_code] = (
        icfilt_epochs
    )

    icfilt_epochs.save(
        icfilt_directory
        / (
            f"icfilt_blk{block_code}_"
            f"{SUBJECT}_{SESSION}-epo.fif"
        ),
        overwrite=True,
    )

    # -------------------------------------------------------------------------
    # Prefix dt
    # -------------------------------------------------------------------------

    dt_epochs = remove_dc_and_linear_trend(
        icfilt_epochs
    )

    dt_blocks[block_code] = (
        dt_epochs
    )

    dt_epochs.save(
        dt_directory
        / (
            f"dt_blk{block_code}_"
            f"{SUBJECT}_{SESSION}-epo.fif"
        ),
        overwrite=True,
    )

    # -------------------------------------------------------------------------
    # Prefix reref
    # -------------------------------------------------------------------------

    reref_epochs = average_rereference(
        dt_epochs
    )

    reref_blocks[block_code] = (
        reref_epochs
    )

    reref_epochs.save(
        reref_directory
        / (
            f"reref_blk{block_code}_"
            f"{SUBJECT}_{SESSION}-epo.fif"
        ),
        overwrite=True,
    )

    # -------------------------------------------------------------------------
    # Prefix bl
    # -------------------------------------------------------------------------

    bl_epochs = apply_baseline_correction(
        reref_epochs
    )

    bl_blocks[block_code] = (
        bl_epochs
    )

    bl_epochs.save(
        bl_directory
        / (
            f"bl_blk{block_code}_"
            f"{SUBJECT}_{SESSION}-epo.fif"
        ),
        overwrite=True,
    )
    
# =============================================================================
# STEP 23 — COMPLETE FINAL EPOCH METADATA
#
# Each final epoch will contain:
#
# subject
# session
# session_number
# block_code
# block_within_session
# cv_block
# trial_within_block
# identity
# race
# original_event_code
# identity_repetition_within_block
# is_oddball
#
# The event code is also retained in epochs.events[:, 2].
# =============================================================================


def complete_epoch_metadata(
    epochs: mne.Epochs,
    subject: str,
    session: str,
) -> mne.Epochs:
    """
    Add subject/session and decoding-related metadata to one block.
    """

    output = epochs.copy()

    if output.metadata is None:

        raise RuntimeError(
            "Epoch metadata is missing."
        )

    metadata = output.metadata.copy()

    session_match = re.fullmatch(
        r"(?:S|session)?(\d+)",
        session,
        re.IGNORECASE,
    )

    if not session_match:

        raise ValueError(
            f"Cannot determine session number from {session!r}."
        )

    session_number = int(
        session_match.group(1)
    )

    metadata["subject"] = subject

    metadata["session"] = session

    metadata["session_number"] = (
        session_number
    )

    metadata["block_within_session"] = (
        metadata["block_code"]
        - 201
    )

    # Creates blocks 1-16 for S1 and 17-32 for S2.
    metadata["cv_block"] = (
        (session_number - 1) * 16
        + metadata["block_within_session"]
    )

    # The identity trigger is the original event code.
    metadata["original_event_code"] = (
        output.events[:, 2]
    )

    metadata["is_oddball"] = False

    output.metadata = metadata

    return output

# =============================================================================
# STEP 24 — CREATE ALL_ID, A_ID, AND C_ID EPOCH FILES
#
# No second epoch extraction is necessary.
#
# The bl files already contain:
#     stimulus codes 1-60
#     epoch start -0.1 s
#     duration 1 s
#
# We concatenate all baseline-corrected blocks and separate epochs using
# the identity/race metadata.
#
# Outputs:
# all_id = identities 1-60
# a_id   = Asian identities 1-30
# c_id   = Caucasian identities 31-60
# =============================================================================


def combine_baseline_blocks(
    bl_blocks: dict[int, mne.Epochs],
    subject: str,
    session: str,
) -> mne.Epochs:
    """
    Add final metadata to every block and concatenate all testing blocks.
    """

    prepared_blocks = []

    for block_code in TEST_BLOCKS:

        prepared = complete_epoch_metadata(
            epochs=bl_blocks[block_code],
            subject=subject,
            session=session,
        )

        prepared_blocks.append(
            prepared
        )

    all_epochs = mne.concatenate_epochs(
        prepared_blocks,
        add_offset=True,
        on_mismatch="raise",
        verbose="INFO",
    )

    return all_epochs


def select_identity_range(
    epochs: mne.Epochs,
    first_identity: int,
    last_identity: int,
) -> mne.Epochs:
    """
    Select epochs according to the identity column in metadata.
    """

    if epochs.metadata is None:

        raise RuntimeError(
            "Metadata is missing."
        )

    selection = epochs.metadata[
        "identity"
    ].between(
        first_identity,
        last_identity,
        inclusive="both",
    )
    return epochs[
        selection.to_numpy()
    ]


# =============================================================================
# CREATE FINAL EPOCH OBJECTS
# =============================================================================

final_directory = (
    output_root
    / "12_final"
)

final_directory.mkdir(
    parents=True,
    exist_ok=True,
)


all_id_epochs = combine_baseline_blocks(
    bl_blocks=bl_blocks,
    subject=SUBJECT,
    session=SESSION,
)

a_id_epochs = select_identity_range(
    epochs=all_id_epochs,
    first_identity=1,
    last_identity=30,
)

c_id_epochs = select_identity_range(
    epochs=all_id_epochs,
    first_identity=31,
    last_identity=60,
)


all_id_path = (
    final_directory
    / (
        f"ALL_id01-60_"
        f"{SUBJECT}_{SESSION}-epo.fif"
    )
)

a_id_path = (
    final_directory
    / (
        f"a_id01-30_"
        f"{SUBJECT}_{SESSION}-epo.fif"
    )
)

c_id_path = (
    final_directory
    / (
        f"c_id31-60_"
        f"{SUBJECT}_{SESSION}-epo.fif"
    )
)


# =============================================================================
# FUNCTION FOR SAVING METADATA
# =============================================================================

def save_metadata_csv(
    epochs: mne.Epochs,
    output_path: Path,
) -> None:
    """
    Save Epochs metadata as a CSV audit table.
    """

    if epochs.metadata is None:

        raise RuntimeError(
            "Cannot save missing metadata."
        )

    epochs.metadata.reset_index(
        drop=True
    ).to_csv(
        output_path,
        index=False,
    )


# =============================================================================
# FINAL VALIDATION FUNCTION
# =============================================================================

def validate_final_epochs(
    epochs: mne.Epochs,
    label: str,
    expected_identity_min: int,
    expected_identity_max: int,
) -> None:

    if epochs.metadata is None:

        raise RuntimeError(
            f"{label}: metadata is missing."
        )

    if len(epochs.metadata) != len(epochs):

        raise RuntimeError(
            f"{label}: metadata rows "
            f"({len(epochs.metadata)}) do not match "
            f"epoch count ({len(epochs)})."
        )

    if len(epochs.events) != len(epochs):

        raise RuntimeError(
            f"{label}: event rows "
            f"({len(epochs.events)}) do not match "
            f"epoch count ({len(epochs)})."
        )

    if not np.array_equal(
        epochs.metadata[
            "original_event_code"
        ].to_numpy(dtype=int),
        epochs.events[:, 2].astype(int),
    ):
        raise RuntimeError(
            f"{label}: original_event_code metadata "
            "does not match epochs.events[:, 2]."
        )

    metadata_identities = set(
        epochs.metadata[
            "identity"
        ].astype(int)
    )

    expected_identities = set(
        range(
            expected_identity_min,
            expected_identity_max + 1,
        )
    )

    unexpected = (
        metadata_identities
        - expected_identities
    )

    missing = (
        expected_identities
        - metadata_identities
    )

    if unexpected:

        raise RuntimeError(
            f"{label}: unexpected identities "
            f"{sorted(unexpected)}."
        )

    if missing:

        raise RuntimeError(
            f"{label}: missing identities "
            f"{sorted(missing)}."
        )

    event_codes = set(
        epochs.events[:, 2].astype(int)
    )

    if not event_codes.issubset(
        expected_identities
    ):

        raise RuntimeError(
            f"{label}: unexpected event codes "
            f"{sorted(event_codes - expected_identities)}."
        )

    print("\n" + "-" * 70)

    print(label)

    print(
        f"Epochs: {len(epochs)}"
    )

    print(
        f"Identity range: "
        f"{epochs.metadata['identity'].min()}-"
        f"{epochs.metadata['identity'].max()}"
    )

    print(
        f"Blocks: "
        f"{sorted(epochs.metadata['block_code'].unique())}"
    )

    print(
        f"Epoch samples: {len(epochs.times)}"
    )

    print(
        f"Epoch time range: "
        f"{epochs.times[0]:.6f} to "
        f"{epochs.times[-1]:.6f} seconds"
    )

    print(
        f"Metadata columns: "
        f"{list(epochs.metadata.columns)}"
    )


# =============================================================================
# VALIDATE FIRST
# =============================================================================

validate_final_epochs(
    epochs=all_id_epochs,
    label="ALL_id",
    expected_identity_min=1,
    expected_identity_max=60,
)

validate_final_epochs(
    epochs=a_id_epochs,
    label="a_id Asian",
    expected_identity_min=1,
    expected_identity_max=30,
)

validate_final_epochs(
    epochs=c_id_epochs,
    label="c_id Caucasian",
    expected_identity_min=31,
    expected_identity_max=60,
)


# =============================================================================
# SAVE ONLY AFTER VALIDATION PASSES
# =============================================================================

print(
    "\nAll final epoch validation checks passed."
)

all_id_epochs.save(
    all_id_path,
    overwrite=True,
)

a_id_epochs.save(
    a_id_path,
    overwrite=True,
)

c_id_epochs.save(
    c_id_path,
    overwrite=True,
)


save_metadata_csv(
    epochs=all_id_epochs,
    output_path=(
        final_directory
        / (
            f"ALL_id01-60_"
            f"{SUBJECT}_{SESSION}_metadata.csv"
        )
    ),
)

save_metadata_csv(
    epochs=a_id_epochs,
    output_path=(
        final_directory
        / (
            f"a_id01-30_"
            f"{SUBJECT}_{SESSION}_metadata.csv"
        )
    ),
)

save_metadata_csv(
    epochs=c_id_epochs,
    output_path=(
        final_directory
        / (
            f"c_id31-60_"
            f"{SUBJECT}_{SESSION}_metadata.csv"
        )
    ),
)

print(
    "\nFinal FIF and metadata files saved successfully."
)


print("\nCompleted prefixes:")

print("ica")

print("icfilt")

print("dt")

print("reref")

print("bl")

print("ALL_id")

print("a_id")

print("c_id")


