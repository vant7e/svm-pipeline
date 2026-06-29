from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
import mne


def read_trial_file(file_path):
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"No 'data' dataset found in {file_path}")
        x = np.array(f["data"], dtype=np.float32)

    x = np.squeeze(x)

    if x.ndim != 3:
        raise ValueError(f"Unexpected squeezed shape {x.shape} in {file_path}")

    if x.shape[0] == 512 and x.shape[1] == 64:
        x = np.transpose(x, (2, 1, 0))
    elif x.shape[1] == 64 and x.shape[2] == 512:
        pass
    else:
        raise ValueError(f"Unexpected data shape {x.shape} in {file_path}")

    return x.astype(np.float32, copy=False)


def parse_face_filename(file_path):
    file_path = Path(file_path)
    name = file_path.name

    id_match = re.search(r"([ac])_id\s+(\d+)", name)
    block_match = re.search(r"trl blk\s+(\d+)", name)

    session_match = re.search(
        r"Session\s*(\d+)",
        file_path.parent.name,
        re.IGNORECASE,
    )

    if session_match is None:
        session_match = re.search(r"_S(\d+)", name)

    if id_match is None or block_match is None or session_match is None:
        return None

    race_code = id_match.group(1)
    identity = int(id_match.group(2))
    block = int(block_match.group(1))
    session = int(session_match.group(1))

    return {
        "race_code": race_code,
        "race": "Asian" if race_code == "a" else "Caucasian",
        "identity": identity,
        "block": block,
        "session": session,
    }


def find_face_subjects(root):
    root = Path(root)
    subjects = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        try:
            children = list(folder.iterdir())
        except PermissionError:
            continue

        has_session_folder = any(
            child.is_dir()
            and re.match(r"Session\s*\d+", child.name, re.IGNORECASE)
            for child in children
        )

        if has_session_folder:
            subjects.append(folder)

    return sorted(set(subjects))


def find_face_trial_files(subject_root):
    subject_root = Path(subject_root)

    files = sorted(
        list(subject_root.glob("Session*/a_id*.mat"))
        + list(subject_root.glob("Session*/c_id*.mat"))
    )

    valid_files = []
    skipped_files = []

    for file in files:
        if parse_face_filename(file) is None:
            skipped_files.append(file)
        else:
            valid_files.append(file)

    return valid_files, skipped_files


def average_repeated_trials(X, metadata):
    """
    Average repetitions within each identity × block × session.
    """
    group_cols = [
        "identity",
        "race_code",
        "race",
        "block",
        "session",
    ]

    X_avg = []
    meta_avg = []

    grouped = metadata.groupby(group_cols, sort=False)

    for _, idx in grouped.groups.items():
        idx = np.asarray(list(idx))

        X_avg.append(X[idx].mean(axis=0))

        row = metadata.iloc[idx[0]].copy()
        row["n_averaged"] = len(idx)
        row["repetition"] = "avg"
        row["trial_mode"] = "averaged"

        meta_avg.append(row)

    X_avg = np.stack(X_avg).astype(np.float32)
    metadata_avg = pd.DataFrame(meta_avg).reset_index(drop=True)

    return X_avg, metadata_avg


def load_face_subject(
    subject_root,
    subject_id=None,
    average_repetitions=False,
    verbose=True,
):
    subject_root = Path(subject_root)
    files, skipped_files = find_face_trial_files(subject_root)

    if verbose:
        print(f"Subject root: {subject_root}")
        print(f"Valid trial files found: {len(files)}")
        print(f"Skipped files: {len(skipped_files)}")

    X_list = []
    meta_rows = []

    for i, file in enumerate(files):
        if verbose and i % 100 == 0:
            print(f"Loading file {i}/{len(files)}")

        info = parse_face_filename(file)
        if info is None:
            continue

        x = read_trial_file(file)

        for rep in range(x.shape[0]):
            X_list.append(x[rep])
            meta_rows.append({
                "subject": subject_id if subject_id is not None else subject_root.name,
                "file": str(file),
                "identity": info["identity"],
                "race_code": info["race_code"],
                "race": info["race"],
                "block": info["block"],
                "session": info["session"],
                "repetition": rep + 1,
                "n_repetitions_in_file": x.shape[0],
                "trial_mode": "single",
            })

    if len(X_list) == 0:
        raise RuntimeError(f"No valid a_id/c_id files found in {subject_root}")

    if verbose:
        print("Stacking trials...")

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    metadata = pd.DataFrame(meta_rows)

    if average_repetitions:
        if verbose:
            print("Averaging repeated trials...")

        X, metadata = average_repeated_trials(X, metadata)

        if verbose:
            print(f"Trials after averaging: {len(metadata)}")

    if len(metadata) != X.shape[0]:
        raise RuntimeError(
            f"Metadata rows ({len(metadata)}) do not match X trials ({X.shape[0]})"
        )

    return X, metadata


def make_eeg_channel_names(n_channels=64):
    return [f"EEG{ch:03d}" for ch in range(1, n_channels + 1)]


def make_epochs_from_array(
    X,
    metadata,
    sfreq=512.0,
    tmin=0.0,
    ch_names=None,
):
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got {X.shape}")

    n_trials, n_channels, _ = X.shape

    if len(metadata) != n_trials:
        raise ValueError(
            f"Metadata rows ({len(metadata)}) do not match X trials ({n_trials})"
        )

    if ch_names is None:
        ch_names = make_eeg_channel_names(n_channels)

    if len(ch_names) != n_channels:
        raise ValueError(
            f"Number of channel names ({len(ch_names)}) does not match "
            f"number of channels ({n_channels})"
        )

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    events = np.column_stack([
        np.arange(n_trials),
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int),
    ])

    epochs = mne.EpochsArray(
        data=X,
        info=info,
        events=events,
        tmin=tmin,
        event_id={"face": 1},
        metadata=metadata.copy(),
        verbose=False,
    )

    return epochs


def load_face_subject_as_epochs(
    subject_root,
    subject_id=None,
    average_repetitions=False,
    sfreq=512.0,
    tmin=0.0,
    ch_names=None,
    verbose=True,
):
    X, metadata = load_face_subject(
        subject_root=subject_root,
        subject_id=subject_id,
        average_repetitions=average_repetitions,
        verbose=verbose,
    )

    epochs = make_epochs_from_array(
        X=X,
        metadata=metadata,
        sfreq=sfreq,
        tmin=tmin,
        ch_names=ch_names,
    )

    return epochs, metadata


def summarize_metadata(metadata):
    print("\n===== Metadata summary =====")
    print("Number of trials:", len(metadata))
    print("Subjects:", metadata["subject"].nunique())
    print("Identities:", metadata["identity"].nunique())

    if "trial_mode" in metadata.columns:
        print("\nTrial mode:")
        print(metadata["trial_mode"].value_counts())

    print("\nRace counts:")
    print(metadata["race"].value_counts())

    print("\nSessions:")
    print(metadata["session"].value_counts().sort_index())

    if "n_repetitions_in_file" in metadata.columns:
        print("\nRepetitions per file:")
        print(
            metadata.drop_duplicates("file")["n_repetitions_in_file"]
            .value_counts()
            .sort_index()
        )

    if "n_averaged" in metadata.columns:
        print("\nTrials averaged per averaged epoch:")
        print(metadata["n_averaged"].value_counts().sort_index())

    print("\nBlocks:")
    print("Number of unique blocks:", metadata["block"].nunique())


def save_face_subject(
    subject_root,
    out_dir,
    subject_id=None,
    average_repetitions=False,
    verbose=True,
):
    X, metadata = load_face_subject(
        subject_root=subject_root,
        subject_id=subject_id,
        average_repetitions=average_repetitions,
        verbose=verbose,
    )

    subject_root = Path(subject_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_name = subject_id if subject_id is not None else subject_root.name
    mode_tag = "avg" if average_repetitions else "single"

    x_path = out_dir / f"{subject_name}_{mode_tag}_X_raw.npy"
    meta_path = out_dir / f"{subject_name}_{mode_tag}_metadata.csv"

    np.save(x_path, X)
    metadata.to_csv(meta_path, index=False)

    print("\n===== Saved =====")
    print(x_path)
    print(meta_path)
    print("X shape:", X.shape)

    summarize_metadata(metadata)

    return X, metadata


def save_face_subject_epochs(
    subject_root,
    out_dir,
    subject_id=None,
    average_repetitions=False,
    sfreq=512.0,
    tmin=0.0,
    ch_names=None,
    verbose=True,
):
    subject_root = Path(subject_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, metadata = load_face_subject_as_epochs(
        subject_root=subject_root,
        subject_id=subject_id,
        average_repetitions=average_repetitions,
        sfreq=sfreq,
        tmin=tmin,
        ch_names=ch_names,
        verbose=verbose,
    )

    subject_name = subject_id if subject_id is not None else subject_root.name
    mode_tag = "avg" if average_repetitions else "single"

    epoch_path = out_dir / f"{subject_name}_{mode_tag}-epo.fif"
    meta_path = out_dir / f"{subject_name}_{mode_tag}_metadata.csv"

    epochs.save(epoch_path, overwrite=True)
    metadata.to_csv(meta_path, index=False)

    print("\n===== Saved Epochs =====")
    print(epoch_path)
    print(meta_path)
    print("Epochs data shape:", epochs.get_data().shape)
    print("Sampling rate:", epochs.info["sfreq"])
    print("Time range:", epochs.times[0], "to", epochs.times[-1])

    summarize_metadata(metadata)

    return epochs, metadata
