# svm/core/time.py

import numpy as np

from .schema import validate_feature_array


VALID_TIME_MODES = {
    "point",
    "window",
    "sliding",
    "full",
    "mean",
}


def get_time_indices(times, tmin=None, tmax=None):
    """
    Return indices for a requested time range.

    Parameters
    ----------
    times : array-like
        Time vector in seconds.
    tmin : float or None
        Start time in seconds.
    tmax : float or None
        End time in seconds.

    Returns
    -------
    idx : np.ndarray
        Time indices within [tmin, tmax].
    """

    times = np.asarray(times)

    if tmin is None:
        tmin = float(times[0])

    if tmax is None:
        tmax = float(times[-1])

    idx = np.where((times >= tmin) & (times <= tmax))[0]

    if len(idx) == 0:
        raise ValueError(
            f"No time points found between tmin={tmin} and tmax={tmax}. "
            f"Available range: {times[0]} to {times[-1]}"
        )

    return idx


def get_nearest_time_index(times, target_time):
    """
    Return index of the closest time point.

    Parameters
    ----------
    times : array-like
        Time vector in seconds.
    target_time : float
        Target time in seconds.

    Returns
    -------
    idx : int
    """

    times = np.asarray(times)

    if target_time < times[0] or target_time > times[-1]:
        raise ValueError(
            f"target_time={target_time} is outside available range "
            f"{times[0]} to {times[-1]}"
        )

    return int(np.argmin(np.abs(times - target_time)))


def make_sliding_windows(times, tmin, tmax, win_ms=10, step_ms=10):
    """
    Build sliding windows in seconds.

    Parameters
    ----------
    times : array-like
        Time vector in seconds.
    tmin : float
        Start time in seconds.
    tmax : float
        End time in seconds.
    win_ms : float
        Window size in milliseconds.
    step_ms : float
        Step size in milliseconds.

    Returns
    -------
    windows : list of dict
        Each dict contains:
            window_id
            t_start
            t_end
            t_center
            indices
            n_timepoints
    """

    times = np.asarray(times)

    win_s = win_ms / 1000.0
    step_s = step_ms / 1000.0

    if win_s <= 0:
        raise ValueError("win_ms must be > 0")

    if step_s <= 0:
        raise ValueError("step_ms must be > 0")

    if tmin >= tmax:
        raise ValueError("tmin must be smaller than tmax")

    windows = []
    start = tmin
    window_id = 0

    while start + win_s <= tmax + 1e-12:
        end = start + win_s
        idx = get_time_indices(times, start, end)

        windows.append({
            "window_id": window_id,
            "t_start": float(start),
            "t_end": float(end),
            "t_center": float((start + end) / 2.0),
            "indices": idx,
            "n_timepoints": int(len(idx)),
        })

        start += step_s
        window_id += 1

    if len(windows) == 0:
        raise ValueError(
            f"No sliding windows created. Check tmin={tmin}, tmax={tmax}, "
            f"win_ms={win_ms}, step_ms={step_ms}"
        )

    return windows


def select_time_slice(
    feature,
    times,
    mode="full",
    tmin=None,
    tmax=None,
    target_time=None,
):
    """
    Select or collapse time dimension from standard feature.

    Input feature shape:
        (freq, time, trial, channel)

    Modes
    -----
    point:
        Select nearest single time point.
        Output shape: (freq, 1, trial, channel)

    window/full:
        Select full time range.
        Output shape: (freq, selected_time, trial, channel)

    mean:
        Average across selected time range.
        Output shape: (freq, 1, trial, channel)

    Parameters
    ----------
    feature : np.ndarray
    times : array-like
    mode : str
    tmin, tmax : float or None
    target_time : float or None

    Returns
    -------
    sliced : np.ndarray
        Time-selected feature.
    info : dict
        Time selection metadata.
    """

    validate_feature_array(feature)

    if mode not in VALID_TIME_MODES:
        raise ValueError(
            f"Invalid mode={mode}. Options: {sorted(VALID_TIME_MODES)}"
        )

    times = np.asarray(times)

    if len(times) != feature.shape[1]:
        raise ValueError(
            f"times length mismatch: len(times)={len(times)}, "
            f"feature time dimension={feature.shape[1]}"
        )

    if mode == "point":
        if target_time is None:
            raise ValueError("target_time must be provided when mode='point'")

        tidx = get_nearest_time_index(times, target_time)

        sliced = feature[:, tidx:tidx + 1, :, :]

        info = {
            "time_mode": "point",
            "target_time": float(target_time),
            "selected_time": float(times[tidx]),
            "time_indices": [int(tidx)],
            "n_timepoints": 1,
        }

        return sliced, info

    if mode in {"window", "full"}:
        tidx = get_time_indices(times, tmin, tmax)

        sliced = feature[:, tidx, :, :]

        info = {
            "time_mode": mode,
            "tmin": float(times[tidx[0]]),
            "tmax": float(times[tidx[-1]]),
            "requested_tmin": tmin,
            "requested_tmax": tmax,
            "time_indices": tidx.astype(int).tolist(),
            "n_timepoints": int(len(tidx)),
        }

        return sliced, info

    if mode == "mean":
        tidx = get_time_indices(times, tmin, tmax)

        sliced = feature[:, tidx, :, :].mean(axis=1, keepdims=True)

        info = {
            "time_mode": "mean",
            "tmin": float(times[tidx[0]]),
            "tmax": float(times[tidx[-1]]),
            "requested_tmin": tmin,
            "requested_tmax": tmax,
            "time_indices": tidx.astype(int).tolist(),
            "n_timepoints_original": int(len(tidx)),
            "n_timepoints_output": 1,
        }

        return sliced, info

    raise RuntimeError(f"Unhandled mode: {mode}")


def feature_to_classifier_matrix(feature, flatten_order="C"):
    """
    Convert standard feature into classifier matrix.

    Input shape:
        (freq, time, trial, channel)

    Output shape:
        X = (trial, features)

    Feature order after transpose:
        trial × freq × time × channel
    then flattened.

    Parameters
    ----------
    feature : np.ndarray
    flatten_order : {"C", "F"}

    Returns
    -------
    X : np.ndarray
        Classifier matrix with shape (trial, features).
    """

    validate_feature_array(feature)

    # (freq, time, trial, channel)
    # -> (trial, freq, time, channel)
    x = np.transpose(feature, (2, 0, 1, 3))

    n_trial = x.shape[0]

    X = x.reshape(n_trial, -1, order=flatten_order)

    return X.astype(np.float32)


def extract_classifier_features(
    feature,
    times,
    mode="full",
    tmin=None,
    tmax=None,
    target_time=None,
    flatten_order="C",
):
    """
    Main helper for SVM scripts.

    Input:
        feature = (freq, time, trial, channel)

    Output:
        X = (trial, features)

    This function supports:
        point
        window
        full
        mean

    For sliding-window decoding, use extract_sliding_classifier_features().
    """

    sliced, time_info = select_time_slice(
        feature=feature,
        times=times,
        mode=mode,
        tmin=tmin,
        tmax=tmax,
        target_time=target_time,
    )

    X = feature_to_classifier_matrix(
        sliced,
        flatten_order=flatten_order,
    )

    info = {
        **time_info,
        "input_shape": list(feature.shape),
        "sliced_shape": list(sliced.shape),
        "classifier_shape": list(X.shape),
        "flatten_order": flatten_order,
    }

    return X, info


def extract_sliding_classifier_features(
    feature,
    times,
    tmin,
    tmax,
    win_ms=10,
    step_ms=10,
    flatten_order="C",
):
    """
    Extract classifier matrices for sliding-window decoding.

    Input:
        feature = (freq, time, trial, channel)

    Output:
        results = list of dict

    Each dict contains:
        window_id
        t_start
        t_end
        t_center
        X
        info

    X shape:
        (trial, features)
    """

    validate_feature_array(feature)

    times = np.asarray(times)

    if len(times) != feature.shape[1]:
        raise ValueError(
            f"times length mismatch: len(times)={len(times)}, "
            f"feature time dimension={feature.shape[1]}"
        )

    windows = make_sliding_windows(
        times=times,
        tmin=tmin,
        tmax=tmax,
        win_ms=win_ms,
        step_ms=step_ms,
    )

    results = []

    for win in windows:
        idx = win["indices"]

        sliced = feature[:, idx, :, :]

        X = feature_to_classifier_matrix(
            sliced,
            flatten_order=flatten_order,
        )

        info = {
            "time_mode": "sliding",
            "window_id": win["window_id"],
            "t_start": win["t_start"],
            "t_end": win["t_end"],
            "t_center": win["t_center"],
            "time_indices": idx.astype(int).tolist(),
            "n_timepoints": win["n_timepoints"],
            "input_shape": list(feature.shape),
            "sliced_shape": list(sliced.shape),
            "classifier_shape": list(X.shape),
            "win_ms": win_ms,
            "step_ms": step_ms,
            "flatten_order": flatten_order,
        }

        results.append({
            "window_id": win["window_id"],
            "t_start": win["t_start"],
            "t_end": win["t_end"],
            "t_center": win["t_center"],
            "X": X,
            "info": info,
        })

    return results


def summarize_time_axis(times):
    """
    Return basic time-axis summary.
    """

    times = np.asarray(times)

    if len(times) < 2:
        step = None
        sfreq_est = None
    else:
        step = float(np.median(np.diff(times)))
        sfreq_est = float(1.0 / step)

    return {
        "n_timepoints": int(len(times)),
        "tmin": float(times[0]),
        "tmax": float(times[-1]),
        "time_step": step,
        "sfreq_est": sfreq_est,
    }


def print_time_summary(times):
    """
    Print readable time-axis summary.
    """

    s = summarize_time_axis(times)

    print("\n==============================")
    print("Time Axis Summary")
    print("==============================")
    print("n_timepoints:", s["n_timepoints"])
    print("tmin:", s["tmin"])
    print("tmax:", s["tmax"])
    print("time_step:", s["time_step"])
    print("sfreq_est:", s["sfreq_est"])
    print("==============================\n")