# svm/core/spatial.py

import re
import numpy as np

from .schema import validate_feature_array


VALID_SELECTION_METHODS = {
    "all",
    "prefix",
    "name",
    "regex",
    "index",
}


def validate_ch_names(ch_names):
    """
    Validate spatial names.

    ch_names can refer to:
        - MEG sensors
        - EEG electrodes
        - source labels
        - ROI names
        - source vertices represented as strings
    """

    if ch_names is None:
        raise ValueError("ch_names cannot be None")

    if len(ch_names) == 0:
        raise ValueError("ch_names is empty")

    return list(ch_names)


def get_spatial_indices(
    ch_names,
    method="all",
    values=None,
    case_sensitive=True,
):
    """
    Generic spatial selector.

    Parameters
    ----------
    ch_names : list[str]
        Names of sensors/electrodes/source labels/ROIs.

    method : str
        One of:
            all
            prefix
            name
            regex
            index

    values : list / str / int / None
        Selection values.

    case_sensitive : bool
        Whether string matching is case-sensitive.

    Returns
    -------
    indices : np.ndarray
        Selected spatial indices.

    selected_names : list[str]
        Names corresponding to selected indices.
    """

    ch_names = validate_ch_names(ch_names)

    if method not in VALID_SELECTION_METHODS:
        raise ValueError(
            f"Invalid selection method: {method}. "
            f"Options: {sorted(VALID_SELECTION_METHODS)}"
        )

    if method == "all":
        indices = np.arange(len(ch_names))

    elif method == "index":
        if values is None:
            raise ValueError("values must be provided for method='index'")

        if isinstance(values, int):
            values = [values]

        indices = np.asarray(values, dtype=int)

        if np.any(indices < 0) or np.any(indices >= len(ch_names)):
            raise IndexError(
                f"Index out of range. Valid range is 0 to {len(ch_names) - 1}"
            )

    elif method == "prefix":
        if values is None:
            raise ValueError("values must be provided for method='prefix'")

        if isinstance(values, str):
            values = [values]

        if not case_sensitive:
            search_values = [v.lower() for v in values]
            search_names = [c.lower() for c in ch_names]
        else:
            search_values = values
            search_names = ch_names

        indices = [
            i for i, name in enumerate(search_names)
            if any(name.startswith(prefix) for prefix in search_values)
        ]

        indices = np.asarray(indices, dtype=int)

    elif method == "name":
        if values is None:
            raise ValueError("values must be provided for method='name'")

        if isinstance(values, str):
            values = [values]

        if not case_sensitive:
            value_set = {v.lower() for v in values}
            indices = [
                i for i, name in enumerate(ch_names)
                if name.lower() in value_set
            ]
        else:
            value_set = set(values)
            indices = [
                i for i, name in enumerate(ch_names)
                if name in value_set
            ]

        indices = np.asarray(indices, dtype=int)

    elif method == "regex":
        if values is None:
            raise ValueError("values must be provided for method='regex'")

        if isinstance(values, str):
            values = [values]

        flags = 0 if case_sensitive else re.IGNORECASE
        patterns = [re.compile(v, flags=flags) for v in values]

        indices = [
            i for i, name in enumerate(ch_names)
            if any(p.search(name) for p in patterns)
        ]

        indices = np.asarray(indices, dtype=int)

    else:
        raise RuntimeError(f"Unhandled method: {method}")

    if len(indices) == 0:
        raise ValueError(
            f"No spatial units selected with method={method}, values={values}"
        )

    selected_names = [ch_names[i] for i in indices]

    return indices, selected_names


def apply_spatial_selection(feature, indices):
    """
    Apply spatial selection to standard feature.

    Input feature shape:
        (freq, time, trial, channel_or_source)

    Output feature shape:
        (freq, time, trial, selected_channel_or_source)
    """

    validate_feature_array(feature)

    indices = np.asarray(indices, dtype=int)

    if np.any(indices < 0) or np.any(indices >= feature.shape[3]):
        raise IndexError(
            f"Spatial index out of range. Feature has {feature.shape[3]} spatial units."
        )

    return feature[:, :, :, indices]


def select_spatial_feature(
    feature,
    ch_names,
    method="all",
    values=None,
    case_sensitive=True,
):
    """
    Select spatial units and return selected feature plus metadata.

    Parameters
    ----------
    feature : np.ndarray
        Standard shape:
            (freq, time, trial, channel_or_source)

    ch_names : list[str]
        Spatial unit names.

    method : str
        all / prefix / name / regex / index

    values : list / str / int / None
        Selection values.

    Returns
    -------
    selected_feature : np.ndarray
    info : dict
    """

    validate_feature_array(feature)

    if len(ch_names) != feature.shape[3]:
        raise ValueError(
            f"ch_names length mismatch: len(ch_names)={len(ch_names)}, "
            f"feature spatial dimension={feature.shape[3]}"
        )

    indices, selected_names = get_spatial_indices(
        ch_names=ch_names,
        method=method,
        values=values,
        case_sensitive=case_sensitive,
    )

    selected_feature = apply_spatial_selection(
        feature=feature,
        indices=indices,
    )

    info = {
        "spatial_method": method,
        "spatial_values": values,
        "case_sensitive": case_sensitive,
        "n_spatial_original": int(feature.shape[3]),
        "n_spatial_selected": int(len(indices)),
        "selected_indices": indices.astype(int).tolist(),
        "selected_names": selected_names,
        "input_shape": list(feature.shape),
        "selected_shape": list(selected_feature.shape),
    }

    return selected_feature, info


def build_ctf_roi_map():
    """
    Default CTF MEG sensor prefix map.

    This is optional convenience only.
    The decoding framework should still accept user-defined selections.
    """

    return {
        "all": {
            "method": "all",
            "values": None,
        },
        "occipital": {
            "method": "prefix",
            "values": ["MLO", "MRO"],
        },
        "temporal": {
            "method": "prefix",
            "values": ["MLT", "MRT"],
        },
        "parietal": {
            "method": "prefix",
            "values": ["MLP", "MRP"],
        },
        "frontal": {
            "method": "prefix",
            "values": ["MLF", "MRF"],
        },
        "occipito_temporal": {
            "method": "prefix",
            "values": ["MLO", "MRO", "MLT", "MRT"],
        },
        "posterior": {
            "method": "prefix",
            "values": ["MLO", "MRO", "MLT", "MRT", "MLP", "MRP"],
        },
        "left_occipito_temporal": {
            "method": "prefix",
            "values": ["MLO", "MLT"],
        },
        "right_occipito_temporal": {
            "method": "prefix",
            "values": ["MRO", "MRT"],
        },
    }


def build_standard_eeg_roi_map():
    """
    Optional EEG electrode selection map.

    This is intentionally simple and can be replaced by user-defined maps.
    """

    return {
        "all": {
            "method": "all",
            "values": None,
        },
        "occipital": {
            "method": "name",
            "values": ["O1", "O2", "Oz", "PO7", "PO8", "POz"],
        },
        "parietal": {
            "method": "name",
            "values": ["P1", "P2", "P3", "P4", "Pz", "P7", "P8"],
        },
        "central": {
            "method": "name",
            "values": ["C1", "C2", "C3", "C4", "Cz"],
        },
        "frontal": {
            "method": "name",
            "values": ["F1", "F2", "F3", "F4", "Fz", "F7", "F8"],
        },
        "frontocentral": {
            "method": "name",
            "values": ["FC1", "FC2", "FC3", "FC4", "FCz"],
        },
    }


def get_selection_from_map(selection_name, selection_map):
    """
    Retrieve a named selection from a user-provided map.

    selection_map example:
        {
            "occipital": {
                "method": "prefix",
                "values": ["MLO", "MRO"]
            }
        }
    """

    if selection_name not in selection_map:
        raise KeyError(
            f"Unknown selection name: {selection_name}. "
            f"Options: {list(selection_map.keys())}"
        )

    selection = selection_map[selection_name]

    if "method" not in selection:
        raise KeyError(f"Selection '{selection_name}' missing key 'method'")

    if "values" not in selection:
        selection["values"] = None

    return selection


def select_by_named_space(
    feature,
    ch_names,
    selection_name,
    selection_map,
    case_sensitive=True,
):
    """
    Select spatial units using a named selection map.

    Works for:
        - CTF MEG sensor ROIs
        - EEG electrode ROIs
        - source labels
        - custom project-specific regions
    """

    selection = get_selection_from_map(
        selection_name=selection_name,
        selection_map=selection_map,
    )

    selected_feature, info = select_spatial_feature(
        feature=feature,
        ch_names=ch_names,
        method=selection["method"],
        values=selection.get("values"),
        case_sensitive=case_sensitive,
    )

    info["selection_name"] = selection_name

    return selected_feature, info


def summarize_spatial_selection(info):
    """
    Human-readable spatial selection summary.
    """

    return (
        f"Spatial selection\n"
        f"-----------------\n"
        f"Selection name: {info.get('selection_name')}\n"
        f"Method: {info['spatial_method']}\n"
        f"Values: {info['spatial_values']}\n"
        f"Original spatial units: {info['n_spatial_original']}\n"
        f"Selected spatial units: {info['n_spatial_selected']}\n"
        f"Selected shape: {info['selected_shape']}\n"
    )


def print_spatial_summary(info):
    """Print spatial selection summary."""
    print(summarize_spatial_selection(info))