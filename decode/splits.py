# svm/decoding/splits.py

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut


def make_loso_splits(n_subjects):
    """
    Leave-one-subject-out splits.

    Use when each subject has one feature matrix:
        X_list = [X_sub1, X_sub2, ...]

    Returns
    -------
    splits : list[dict]
        Each split contains train_subjects and test_subject.
    """

    splits = []

    for test_i in range(n_subjects):
        train_i = [i for i in range(n_subjects) if i != test_i]

        splits.append({
            "split_id": test_i,
            "split_type": "loso",
            "train_subjects": train_i,
            "test_subjects": [test_i],
        })

    return splits


def make_leave_one_group_out_splits(groups):
    """
    Generic leave-one-group-out split.

    Useful for:
        - leave-one-run-out
        - leave-one-block-out
        - leave-one-observation-out
        - repeated-stimulus designs

    Parameters
    ----------
    groups : array-like
        One group label per trial.

    Returns
    -------
    splits : list[dict]
        Each split contains train_idx and test_idx.
    """

    groups = np.asarray(groups)

    logo = LeaveOneGroupOut()
    dummy_X = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))

    splits = []

    for split_id, (train_idx, test_idx) in enumerate(
        logo.split(dummy_X, dummy_y, groups)
    ):
        splits.append({
            "split_id": split_id,
            "split_type": "leave_one_group_out",
            "left_out_group": groups[test_idx][0],
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return splits


def make_leave_one_run_out_splits(run_labels):
    """
    Leave-one-run-out split.

    Wrapper around leave-one-group-out.
    """

    splits = make_leave_one_group_out_splits(run_labels)

    for s in splits:
        s["split_type"] = "leave_one_run_out"
        s["left_out_run"] = s.pop("left_out_group")

    return splits


def make_leave_one_block_out_splits(block_labels):
    """
    Leave-one-block-out split.

    Intended for repeated-block experimental designs.
    """

    splits = make_leave_one_group_out_splits(block_labels)

    for s in splits:
        s["split_type"] = "leave_one_block_out"
        s["left_out_block"] = s.pop("left_out_group")

    return splits


def make_leave_one_observation_out_splits(
    stimulus_labels,
    observation_labels,
):
    """
    Leave-one-observation-out split for repeated-stimulus designs.

    Intended for experiments where the same stimulus appears
    repeatedly across observations/blocks.

    Example structure
    -----------------
        60 faces × 32 observations

    For each left-out observation:
        train = all remaining observations
        test = left-out observation

    This design is commonly used in repeated-observation
    neural decoding analyses.

    Parameters
    ----------
    stimulus_labels : array-like
        Stimulus identity per trial.

    observation_labels : array-like
        Observation/block label per trial.

    Returns
    -------
    splits : list[dict]
    """

    stimulus_labels = np.asarray(stimulus_labels)
    observation_labels = np.asarray(observation_labels)

    if len(stimulus_labels) != len(observation_labels):
        raise ValueError(
            "stimulus_labels and observation_labels must have the same length."
        )

    unique_obs = np.unique(observation_labels)

    splits = []

    for split_id, obs in enumerate(unique_obs):

        test_idx = np.where(observation_labels == obs)[0]
        train_idx = np.where(observation_labels != obs)[0]

        train_stimuli = set(stimulus_labels[train_idx])
        test_stimuli = set(stimulus_labels[test_idx])

        missing_in_train = sorted(test_stimuli - train_stimuli)

        if len(missing_in_train) > 0:
            raise ValueError(
                f"Observation {obs} contains stimuli not present "
                f"in training observations: {missing_in_train[:10]}"
            )

        splits.append({
            "split_id": split_id,
            "split_type": "leave_one_observation_out",
            "left_out_observation": obs,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return splits


def make_stratified_kfold_splits(
    y,
    n_splits=5,
    shuffle=True,
    seed=42,
):
    """
    Stratified K-fold split for trial-level classification.

    Use when trials are independent and labels are available.
    """

    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )

    dummy_X = np.zeros((len(y), 1))

    splits = []

    for split_id, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y)):
        splits.append({
            "split_id": split_id,
            "split_type": "stratified_kfold",
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return splits


def validate_split_indices(n_trials, splits):
    """
    Validate train/test indices for trial-level splits.
    """

    for s in splits:

        if "train_idx" not in s or "test_idx" not in s:
            continue

        train_idx = np.asarray(s["train_idx"])
        test_idx = np.asarray(s["test_idx"])

        if np.any(train_idx < 0) or np.any(train_idx >= n_trials):
            raise IndexError(
                f"Train index out of range in split {s['split_id']}"
            )

        if np.any(test_idx < 0) or np.any(test_idx >= n_trials):
            raise IndexError(
                f"Test index out of range in split {s['split_id']}"
            )

        overlap = set(train_idx).intersection(set(test_idx))

        if len(overlap) > 0:
            raise ValueError(
                f"Train/test overlap found in split {s['split_id']}"
            )

    return True


def apply_trial_split(X, y, split):
    """
    Apply a trial-level split to X and y.

    Parameters
    ----------
    X : np.ndarray
        Shape:
            (trial, features)

    y : np.ndarray
        Shape:
            (trial,)
    """

    train_idx = np.asarray(split["train_idx"])
    test_idx = np.asarray(split["test_idx"])

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )


def describe_splits(splits):
    """
    Return basic split summary.
    """

    if len(splits) == 0:
        return {
            "n_splits": 0,
            "split_type": None,
        }

    split_type = splits[0].get("split_type", "unknown")

    summary = {
        "n_splits": len(splits),
        "split_type": split_type,
    }

    if "train_idx" in splits[0]:
        summary["train_sizes"] = [
            len(s["train_idx"]) for s in splits
        ]

        summary["test_sizes"] = [
            len(s["test_idx"]) for s in splits
        ]

    if "train_subjects" in splits[0]:
        summary["train_subject_counts"] = [
            len(s["train_subjects"]) for s in splits
        ]

        summary["test_subject_counts"] = [
            len(s["test_subjects"]) for s in splits
        ]

    return summary


def print_split_summary(splits):
    """
    Print readable split summary.
    """

    s = describe_splits(splits)

    print("\n==============================")
    print("Split Summary")
    print("==============================")

    print("split_type:", s["split_type"])
    print("n_splits:", s["n_splits"])

    if "train_sizes" in s:
        print("train sizes:", s["train_sizes"][:10])
        print("test sizes:", s["test_sizes"][:10])

    if "train_subject_counts" in s:
        print(
            "train subject counts:",
            s["train_subject_counts"][:10]
        )

        print(
            "test subject counts:",
            s["test_subject_counts"][:10]
        )

    print("==============================\n")