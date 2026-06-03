# svm/core/labels.py

import itertools
import numpy as np
import pandas as pd


def load_metadata(path):
    """Load trial-level metadata CSV."""
    return pd.read_csv(path)


def validate_metadata_length(metadata, n_trials):
    """Check metadata rows match feature trial dimension."""
    if len(metadata) != n_trials:
        raise ValueError(
            f"Metadata length mismatch: metadata has {len(metadata)} rows, "
            f"but feature has {n_trials} trials."
        )


def require_columns(metadata, columns):
    """Ensure required columns exist."""
    missing = [c for c in columns if c not in metadata.columns]
    if missing:
        raise KeyError(f"Missing metadata columns: {missing}")


def get_column_values(metadata, column, as_str=False):
    """Return values from one metadata column."""
    require_columns(metadata, [column])
    values = metadata[column].values
    if as_str:
        values = values.astype(str)
    return values


def build_binary_labels(
    metadata,
    column,
    positive_value,
    negative_value,
    positive_label=1,
    negative_label=0,
    drop_other=True,
):
    """
    Build binary labels from any metadata column.

    Example
    -------
    OLD_NEW:
        positive_value="Target"
        negative_value="Foil"

    Confidence:
        positive_value=4
        negative_value=1
    """

    require_columns(metadata, [column])

    values = metadata[column].values

    pos_mask = values == positive_value
    neg_mask = values == negative_value

    if drop_other:
        keep_mask = pos_mask | neg_mask
    else:
        if (~(pos_mask | neg_mask)).any():
            raise ValueError(
                f"Column {column} contains values other than "
                f"{positive_value} and {negative_value}."
            )
        keep_mask = np.ones(len(values), dtype=bool)

    y = np.full(len(values), fill_value=-1, dtype=int)
    y[pos_mask] = positive_label
    y[neg_mask] = negative_label

    return y[keep_mask], keep_mask


def build_multiclass_labels(
    metadata,
    column,
    class_values=None,
    drop_other=True,
):
    """
    Build multiclass integer labels from any metadata column.

    Example
    -------
    column="item_category"
    column="Confidence"
    column="run"
    """

    require_columns(metadata, [column])

    values = metadata[column].values

    if class_values is None:
        class_values = sorted(pd.unique(values).tolist())

    value_to_label = {v: i for i, v in enumerate(class_values)}

    y = np.full(len(values), fill_value=-1, dtype=int)
    keep_mask = np.zeros(len(values), dtype=bool)

    for v, label in value_to_label.items():
        mask = values == v
        y[mask] = label
        keep_mask |= mask

    if not drop_other and (~keep_mask).any():
        bad_values = sorted(pd.unique(values[~keep_mask]).tolist())
        raise ValueError(f"Unexpected values in {column}: {bad_values}")

    return y[keep_mask], keep_mask, value_to_label


def filter_metadata(metadata, query=None, mask=None):
    """
    Filter metadata by pandas query or boolean mask.

    Example
    -------
    filter_metadata(meta, query="RT > 200")
    filter_metadata(meta, query="Confidence >= 3")
    """

    if query is not None and mask is not None:
        raise ValueError("Use either query or mask, not both.")

    if query is not None:
        return metadata.query(query).reset_index(drop=True)

    if mask is not None:
        return metadata.loc[mask].reset_index(drop=True)

    return metadata.reset_index(drop=True)


def build_pairs_from_column(
    metadata,
    column,
    unique_only=True,
):
    """
    Build all pair combinations from values in a metadata column.

    Example
    -------
    column="image"          object identity pairs
    column="item_category"  category pairs
    column="Confidence"     confidence-level pairs
    """

    require_columns(metadata, [column])

    values = metadata[column].astype(str).tolist()

    if unique_only:
        items = list(dict.fromkeys(values))
    else:
        items = values

    pairs = list(itertools.combinations(range(len(items)), 2))

    pair_table = pd.DataFrame({
        "pair_id": np.arange(len(pairs)),
        "idx_a": [a for a, b in pairs],
        "idx_b": [b for a, b in pairs],
        "value_a": [items[a] for a, b in pairs],
        "value_b": [items[b] for a, b in pairs],
        "pair_column": column,
    })

    return pair_table


def sample_pairs(
    pair_table,
    n_pairs,
    seed=42,
):
    """Randomly sample rows from a pair table."""

    if n_pairs is None or n_pairs >= len(pair_table):
        return pair_table.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pair_table), size=n_pairs, replace=False)

    return pair_table.iloc[idx].reset_index(drop=True)


def chunk_pairs(
    pair_table,
    pair_start=0,
    pair_end=None,
):
    """Select pair chunk for SLURM array jobs."""

    if pair_end is None:
        pair_end = len(pair_table)

    if pair_start < 0 or pair_start >= len(pair_table):
        raise ValueError(
            f"pair_start must be in [0, {len(pair_table)-1}], got {pair_start}"
        )

    if pair_end <= pair_start:
        raise ValueError("pair_end must be greater than pair_start")

    pair_end = min(pair_end, len(pair_table))

    return pair_table.iloc[pair_start:pair_end].reset_index(drop=True)


def validate_trial_alignment(
    metadata_list,
    column,
    strict=True,
):
    """
    Check whether multiple subjects have the same trial order
    for a given metadata column.

    Useful for LOSO decoding where trial/image order must match.
    """

    if len(metadata_list) < 2:
        return True

    ref = metadata_list[0][column].astype(str).tolist()

    for i, meta in enumerate(metadata_list[1:], start=1):
        current = meta[column].astype(str).tolist()

        if strict:
            if current != ref:
                raise ValueError(
                    f"Trial alignment mismatch at metadata index {i} "
                    f"for column '{column}'."
                )
        else:
            if set(current) != set(ref):
                raise ValueError(
                    f"Trial set mismatch at metadata index {i} "
                    f"for column '{column}'."
                )

    return True


def get_trial_indices_by_value(metadata, column, value):
    """Return trial indices where metadata[column] == value."""
    require_columns(metadata, [column])
    return np.where(metadata[column].values == value)[0]


def describe_column(metadata, column):
    """Print counts for one metadata column."""
    require_columns(metadata, [column])
    return metadata[column].value_counts(dropna=False)