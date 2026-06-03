# svm/decoding/projection.py

import numpy as np

from svm.core.schema import validate_feature_array
from svm.core.spatial import (
    select_spatial_feature,
    select_by_named_space,
)
from svm.core.time import (
    extract_classifier_features,
    extract_sliding_classifier_features,
)


VALID_PROJECTION_MODES = {
    "point",
    "window",
    "full",
    "mean",
    "sliding",
}


def project_feature_to_X(
    feature,
    meta,
    time_mode="full",
    tmin=None,
    tmax=None,
    target_time=None,
    spatial_method="all",
    spatial_values=None,
    spatial_selection_name=None,
    spatial_selection_map=None,
    case_sensitive=True,
    win_ms=10,
    step_ms=10,
    flatten_order="C",
):
    """
    Project standard feature into classifier-ready matrix.

    Input feature shape:
        (freq, time, trial, channel_or_source)

    Output for non-sliding modes:
        X: (trial, features)
        info: dict

    Output for sliding mode:
        windows: list of dict
            each item contains:
                X: (trial, features)
                t_start
                t_end
                t_center
                info

    Parameters
    ----------
    feature : np.ndarray
        Standard feature array.

    meta : dict
        Metadata created by core.schema/build_feature_meta.

    time_mode : str
        point / window / full / mean / sliding

    spatial_method : str
        all / prefix / name / regex / index

    spatial_selection_name : str or None
        Optional named spatial selection, e.g. "occipito_temporal".

    spatial_selection_map : dict or None
        Optional map for named selections.

    Notes
    -----
    This function does NOT select labels.
    It only projects neural features into X.
    """

    validate_feature_array(feature)

    if time_mode not in VALID_PROJECTION_MODES:
        raise ValueError(
            f"Invalid time_mode={time_mode}. "
            f"Options: {sorted(VALID_PROJECTION_MODES)}"
        )

    if "times" not in meta:
        raise KeyError("meta must contain 'times'")

    if "ch_names" not in meta:
        raise KeyError("meta must contain 'ch_names'")

    times = np.asarray(meta["times"])
    ch_names = meta["ch_names"]

    # -------------------------
    # Spatial selection
    # -------------------------
    if spatial_selection_name is not None:
        if spatial_selection_map is None:
            raise ValueError(
                "spatial_selection_map must be provided when "
                "spatial_selection_name is used."
            )

        selected_feature, spatial_info = select_by_named_space(
            feature=feature,
            ch_names=ch_names,
            selection_name=spatial_selection_name,
            selection_map=spatial_selection_map,
            case_sensitive=case_sensitive,
        )

    else:
        selected_feature, spatial_info = select_spatial_feature(
            feature=feature,
            ch_names=ch_names,
            method=spatial_method,
            values=spatial_values,
            case_sensitive=case_sensitive,
        )

    # -------------------------
    # Time projection
    # -------------------------
    if time_mode == "sliding":
        windows = extract_sliding_classifier_features(
            feature=selected_feature,
            times=times,
            tmin=tmin,
            tmax=tmax,
            win_ms=win_ms,
            step_ms=step_ms,
            flatten_order=flatten_order,
        )

        for w in windows:
            w["info"]["spatial_info"] = spatial_info
            w["info"]["projection_mode"] = "sliding"

        return windows

    X, time_info = extract_classifier_features(
        feature=selected_feature,
        times=times,
        mode=time_mode,
        tmin=tmin,
        tmax=tmax,
        target_time=target_time,
        flatten_order=flatten_order,
    )

    info = {
        "projection_mode": time_mode,
        "feature_space": meta.get("feature_space"),
        "representation": meta.get("representation"),
        "input_shape": list(feature.shape),
        "spatial_info": spatial_info,
        "time_info": time_info,
        "X_shape": list(X.shape),
    }

    return X, info


def project_multiple_subjects_to_X(
    feature_list,
    meta_list,
    **projection_kwargs,
):
    """
    Apply the same projection to multiple subjects.

    Parameters
    ----------
    feature_list : list[np.ndarray]
    meta_list : list[dict]

    Returns
    -------
    X_list : list[np.ndarray]
        One X matrix per subject.

    info_list : list[dict]
        Projection info per subject.

    Notes
    -----
    This is useful for LOSO decoding:
        subject -> X matrix
    """

    if len(feature_list) != len(meta_list):
        raise ValueError(
            f"feature_list and meta_list length mismatch: "
            f"{len(feature_list)} vs {len(meta_list)}"
        )

    X_list = []
    info_list = []

    for feature, meta in zip(feature_list, meta_list):
        result = project_feature_to_X(
            feature=feature,
            meta=meta,
            **projection_kwargs,
        )

        if isinstance(result, list):
            raise ValueError(
                "project_multiple_subjects_to_X does not support "
                "time_mode='sliding'. Use project_multiple_subjects_sliding_to_X instead."
            )

        X, info = result
        X_list.append(X)
        info_list.append(info)

    return X_list, info_list


def project_multiple_subjects_sliding_to_X(
    feature_list,
    meta_list,
    **projection_kwargs,
):
    """
    Apply sliding-window projection to multiple subjects.

    Returns
    -------
    window_results : list[dict]

    Each item:
        {
            "window_id": int,
            "t_start": float,
            "t_end": float,
            "t_center": float,
            "X_list": list[np.ndarray],
            "info_list": list[dict],
        }
    """

    if len(feature_list) != len(meta_list):
        raise ValueError(
            f"feature_list and meta_list length mismatch: "
            f"{len(feature_list)} vs {len(meta_list)}"
        )

    projection_kwargs = dict(projection_kwargs)
    projection_kwargs["time_mode"] = "sliding"

    all_subject_windows = []

    for feature, meta in zip(feature_list, meta_list):
        windows = project_feature_to_X(
            feature=feature,
            meta=meta,
            **projection_kwargs,
        )
        all_subject_windows.append(windows)

    n_windows = len(all_subject_windows[0])

    for subj_i, windows in enumerate(all_subject_windows):
        if len(windows) != n_windows:
            raise ValueError(
                f"Subject {subj_i} has {len(windows)} windows, "
                f"expected {n_windows}"
            )

    window_results = []

    for w_i in range(n_windows):
        ref = all_subject_windows[0][w_i]

        X_list = []
        info_list = []

        for subj_windows in all_subject_windows:
            w = subj_windows[w_i]

            if w["window_id"] != ref["window_id"]:
                raise ValueError("Sliding window alignment mismatch.")

            X_list.append(w["X"])
            info_list.append(w["info"])

        window_results.append({
            "window_id": ref["window_id"],
            "t_start": ref["t_start"],
            "t_end": ref["t_end"],
            "t_center": ref["t_center"],
            "X_list": X_list,
            "info_list": info_list,
        })

    return window_results


def subset_X_by_trials(X, trial_indices):
    """
    Select rows from classifier matrix.

    X shape:
        (trial, features)
    """

    trial_indices = np.asarray(trial_indices, dtype=int)

    if np.any(trial_indices < 0) or np.any(trial_indices >= X.shape[0]):
        raise IndexError(
            f"Trial index out of range. X has {X.shape[0]} trials."
        )

    return X[trial_indices]


def check_X_list_alignment(X_list):
    """
    Check that multiple subjects have compatible X shapes.

    Allows different number of trials only if explicitly handled later.
    For most LOSO object-identity decoding, all subjects should have
    the same number of trials/images.
    """

    if len(X_list) == 0:
        raise ValueError("X_list is empty")

    ref_shape = X_list[0].shape

    for i, X in enumerate(X_list[1:], start=1):
        if X.shape != ref_shape:
            raise ValueError(
                f"X shape mismatch at index {i}: "
                f"{X.shape} vs reference {ref_shape}"
            )

    return True


def summarize_projection_info(info):
    """
    Human-readable projection summary.
    """

    spatial = info.get("spatial_info", {})
    time = info.get("time_info", {})

    return (
        "Projection summary\n"
        "------------------\n"
        f"Feature space: {info.get('feature_space')}\n"
        f"Representation: {info.get('representation')}\n"
        f"Projection mode: {info.get('projection_mode')}\n"
        f"Input shape: {info.get('input_shape')}\n"
        f"X shape: {info.get('X_shape')}\n"
        f"Spatial method: {spatial.get('spatial_method')}\n"
        f"Spatial selected: {spatial.get('n_spatial_selected')}\n"
        f"Time mode: {time.get('time_mode')}\n"
        f"Time points: {time.get('n_timepoints', time.get('n_timepoints_output'))}\n"
    )


def print_projection_summary(info):
    """Print readable projection summary."""
    print(summarize_projection_info(info))