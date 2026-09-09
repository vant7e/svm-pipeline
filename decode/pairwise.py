# svm/decode/pairwise.py

from __future__ import annotations

import numpy as np
import pandas as pd

from joblib import (
    Parallel,
    delayed,
)

from svm.decode.classifiers import (
    build_classifier,
    fit_classifier,
    predict_classifier,
    decision_values,
    classifier_config_dict,
)

from svm.decode.metrics import (
    DEFAULT_METRICS,
    compute_fold_result,
    aggregate_fold_results,
)

from svm.decode.splits import (
    make_loso_splits,
)


# =============================================================================
# HELPERS
# =============================================================================


def _requires_decision_values(
    metrics,
):
    """
    Return whether requested metrics require classifier decision scores.
    """

    requested_metrics = (
        DEFAULT_METRICS
        if metrics is None
        else metrics
    )

    return (
        "auc"
        in requested_metrics
    )


def get_pair_trial_indices(
    values,
    value_a,
    value_b,
):
    """
    Return global observation indices and binary labels for one pair.
    """

    values = np.asarray(
        values
    )

    mask_a = (
        values
        == value_a
    )

    mask_b = (
        values
        == value_b
    )

    idx_a = np.where(
        mask_a
    )[0]

    idx_b = np.where(
        mask_b
    )[0]

    if len(
        idx_a
    ) == 0:

        raise ValueError(
            f"No trials found for "
            f"value_a={value_a}"
        )

    if len(
        idx_b
    ) == 0:

        raise ValueError(
            f"No trials found for "
            f"value_b={value_b}"
        )

    idx = np.concatenate(
        [
            idx_a,
            idx_b,
        ]
    )

    y = np.concatenate(
        [
            np.zeros(
                len(
                    idx_a
                ),
                dtype=int,
            ),

            np.ones(
                len(
                    idx_b
                ),
                dtype=int,
            ),
        ]
    )

    return (
        idx,
        y,
    )


# =============================================================================
# PAIR PLAN
# =============================================================================


def build_single_subject_pair_plan(
    values,
    pair_table,
    splits,
):
    """
    Precompute all pair-specific CV train/test indices.

    This plan depends only on:

        pair identities
        observation labels
        CV splits

    and therefore can be reused for every frequency/time/window.
    """

    values = np.asarray(
        values
    )

    required_cols = {
        "value_a",
        "value_b",
    }

    if not required_cols.issubset(
        pair_table.columns
    ):

        raise KeyError(
            "pair_table must contain "
            f"columns: {required_cols}"
        )

    pair_plan = []

    for row_i, row in pair_table.iterrows():

        value_a = row[
            "value_a"
        ]

        value_b = row[
            "value_b"
        ]

        (
            pair_idx,
            pair_y,
        ) = get_pair_trial_indices(
            values=values,
            value_a=value_a,
            value_b=value_b,
        )

        fold_plan = []

        for split in splits:

            train_idx_global = np.asarray(
                split[
                    "train_idx"
                ],
                dtype=int,
            )

            test_idx_global = np.asarray(
                split[
                    "test_idx"
                ],
                dtype=int,
            )

            train_mask = np.isin(
                pair_idx,
                train_idx_global,
            )

            test_mask = np.isin(
                pair_idx,
                test_idx_global,
            )

            train_pair_idx = pair_idx[
                train_mask
            ]

            test_pair_idx = pair_idx[
                test_mask
            ]

            y_train = pair_y[
                train_mask
            ]

            y_test = pair_y[
                test_mask
            ]

            if len(
                np.unique(
                    y_train
                )
            ) < 2:
                continue

            if len(
                np.unique(
                    y_test
                )
            ) < 2:
                continue

            fold_plan.append(
                {
                    "split_id": split[
                        "split_id"
                    ],

                    "split_type": split.get(
                        "split_type"
                    ),

                    "train_pair_idx": (
                        train_pair_idx
                    ),

                    "test_pair_idx": (
                        test_pair_idx
                    ),

                    "y_train": (
                        y_train
                    ),

                    "y_test": (
                        y_test
                    ),
                }
            )

        pair_plan.append(
            {
                "pair_index": int(
                    row_i
                ),

                "value_a": (
                    value_a
                ),

                "value_b": (
                    value_b
                ),

                "row": (
                    row
                ),

                "folds": (
                    fold_plan
                ),
            }
        )

    return pair_plan


# =============================================================================
# SINGLE PAIR: STANDARD SPLIT INPUT
# =============================================================================


def decode_pair_single_subject_cv(
    X,
    values,
    value_a,
    value_b,
    splits,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    pca_components=None,
):
    """
    Decode one pair using trial-level CV.

    StandardScaler and optional PCA are both fitted separately inside each
    training fold through the sklearn Pipeline.
    """

    values = np.asarray(
        values
    )

    (
        pair_idx,
        pair_y,
    ) = get_pair_trial_indices(
        values=values,
        value_a=value_a,
        value_b=value_b,
    )

    fold_results = []

    for split in splits:

        train_idx_global = np.asarray(
            split[
                "train_idx"
            ]
        )

        test_idx_global = np.asarray(
            split[
                "test_idx"
            ]
        )

        train_mask = np.isin(
            pair_idx,
            train_idx_global,
        )

        test_mask = np.isin(
            pair_idx,
            test_idx_global,
        )

        train_pair_idx = pair_idx[
            train_mask
        ]

        test_pair_idx = pair_idx[
            test_mask
        ]

        y_train = pair_y[
            train_mask
        ]

        y_test = pair_y[
            test_mask
        ]

        if len(
            np.unique(
                y_train
            )
        ) < 2:
            continue

        if len(
            np.unique(
                y_test
            )
        ) < 2:
            continue

        X_train = X[
            train_pair_idx
        ]

        X_test = X[
            test_pair_idx
        ]

        clf = build_classifier(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            pca_components=pca_components,
        )

        clf = fit_classifier(
            clf,
            X_train,
            y_train,
        )

        y_pred = predict_classifier(
            clf,
            X_test,
        )

        if _requires_decision_values(
            metrics
        ):

            y_score = decision_values(
                clf,
                X_test,
            )

        else:

            y_score = None

        fold_result = compute_fold_result(
            split_id=split[
                "split_id"
            ],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            average="binary",
            extra_info={
                "value_a": value_a,
                "value_b": value_b,
                "n_train": int(
                    len(
                        y_train
                    )
                ),
                "n_test": int(
                    len(
                        y_test
                    )
                ),
                "split_type": split.get(
                    "split_type"
                ),
            },
        )

        fold_results.append(
            fold_result
        )

    if len(
        fold_results
    ) == 0:

        raise ValueError(
            f"No valid folds for pair "
            f"{value_a} vs {value_b}."
        )

    summary = aggregate_fold_results(
        fold_results
    )

    return {
        "value_a": value_a,
        "value_b": value_b,
        "fold_results": fold_results,
        "summary": summary,
    }


# =============================================================================
# SINGLE PAIR: PRECOMPUTED PLAN
# =============================================================================


def decode_pair_single_subject_cv_from_plan(
    X,
    pair_entry,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    pca_components=None,
):
    """
    Decode one pair from the precomputed pair/fold plan.
    """

    value_a = pair_entry[
        "value_a"
    ]

    value_b = pair_entry[
        "value_b"
    ]

    fold_results = []

    for fold in pair_entry[
        "folds"
    ]:

        train_pair_idx = fold[
            "train_pair_idx"
        ]

        test_pair_idx = fold[
            "test_pair_idx"
        ]

        y_train = fold[
            "y_train"
        ]

        y_test = fold[
            "y_test"
        ]

        X_train = X[
            train_pair_idx
        ]

        X_test = X[
            test_pair_idx
        ]

        clf = build_classifier(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            pca_components=pca_components,
        )

        clf = fit_classifier(
            clf,
            X_train,
            y_train,
        )

        y_pred = predict_classifier(
            clf,
            X_test,
        )

        if _requires_decision_values(
            metrics
        ):

            y_score = decision_values(
                clf,
                X_test,
            )

        else:

            y_score = None

        fold_result = compute_fold_result(
            split_id=fold[
                "split_id"
            ],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            average="binary",
            extra_info={
                "value_a": value_a,
                "value_b": value_b,
                "n_train": int(
                    len(
                        y_train
                    )
                ),
                "n_test": int(
                    len(
                        y_test
                    )
                ),
                "split_type": fold[
                    "split_type"
                ],
            },
        )

        fold_results.append(
            fold_result
        )

    if len(
        fold_results
    ) == 0:

        raise ValueError(
            f"No valid folds for pair "
            f"{value_a} vs {value_b}."
        )

    summary = aggregate_fold_results(
        fold_results
    )

    return {
        "value_a": value_a,
        "value_b": value_b,
        "fold_results": fold_results,
        "summary": summary,
    }


# =============================================================================
# LOSO
# =============================================================================


def decode_pair_loso(
    X_list,
    values_list,
    value_a,
    value_b,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    pca_components=None,
):
    """
    Decode one pair using leave-one-subject-out.
    """

    if len(
        X_list
    ) != len(
        values_list
    ):

        raise ValueError(
            "X_list and values_list "
            "must have the same length."
        )

    n_subjects = len(
        X_list
    )

    splits = make_loso_splits(
        n_subjects
    )

    fold_results = []

    for split in splits:

        train_subjects = split[
            "train_subjects"
        ]

        test_subject = split[
            "test_subjects"
        ][
            0
        ]

        X_train_all = []
        y_train_all = []

        for si in train_subjects:

            idx, y = get_pair_trial_indices(
                values=values_list[
                    si
                ],
                value_a=value_a,
                value_b=value_b,
            )

            X_train_all.append(
                X_list[
                    si
                ][
                    idx
                ]
            )

            y_train_all.append(
                y
            )

        X_train = np.vstack(
            X_train_all
        )

        y_train = np.concatenate(
            y_train_all
        )

        (
            test_idx,
            y_test,
        ) = get_pair_trial_indices(
            values=values_list[
                test_subject
            ],
            value_a=value_a,
            value_b=value_b,
        )

        X_test = X_list[
            test_subject
        ][
            test_idx
        ]

        if len(
            np.unique(
                y_train
            )
        ) < 2:
            continue

        if len(
            np.unique(
                y_test
            )
        ) < 2:
            continue

        clf = build_classifier(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            pca_components=pca_components,
        )

        clf = fit_classifier(
            clf,
            X_train,
            y_train,
        )

        y_pred = predict_classifier(
            clf,
            X_test,
        )

        if _requires_decision_values(
            metrics
        ):

            y_score = decision_values(
                clf,
                X_test,
            )

        else:

            y_score = None

        fold_result = compute_fold_result(
            split_id=split[
                "split_id"
            ],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            average="binary",
            extra_info={
                "value_a": value_a,
                "value_b": value_b,
                "n_train": int(
                    len(
                        y_train
                    )
                ),
                "n_test": int(
                    len(
                        y_test
                    )
                ),
                "split_type": "loso",
                "test_subject": int(
                    test_subject
                ),
            },
        )

        fold_results.append(
            fold_result
        )

    if len(
        fold_results
    ) == 0:

        raise ValueError(
            f"No valid LOSO folds for pair "
            f"{value_a} vs {value_b}."
        )

    summary = aggregate_fold_results(
        fold_results
    )

    return {
        "value_a": value_a,
        "value_b": value_b,
        "fold_results": fold_results,
        "summary": summary,
    }


def run_pairwise_loso_decoding(
    X_list,
    values_list,
    pair_table,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    verbose=True,
    pca_components=None,
):
    """
    Run pairwise LOSO decoding.
    """

    required_cols = {
        "value_a",
        "value_b",
    }

    if not required_cols.issubset(
        pair_table.columns
    ):

        raise KeyError(
            "pair_table must contain "
            f"columns: {required_cols}"
        )

    pair_rows = []
    fold_rows = []

    for row_i, row in pair_table.iterrows():

        value_a = row[
            "value_a"
        ]

        value_b = row[
            "value_b"
        ]

        if verbose:

            print(
                f"[PAIR {row_i + 1}/"
                f"{len(pair_table)}] "
                f"{value_a} vs {value_b}",
                flush=True,
            )

        try:

            result = decode_pair_loso(
                X_list=X_list,
                values_list=values_list,
                value_a=value_a,
                value_b=value_b,
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                metrics=metrics,
                pca_components=pca_components,
            )

            summary = result[
                "summary"
            ]

            pair_row = {
                "pair_index": int(
                    row_i
                ),
                "value_a": value_a,
                "value_b": value_b,
                "status": "ok",
            }

            for col in pair_table.columns:

                if col not in pair_row:

                    pair_row[
                        col
                    ] = row[
                        col
                    ]

            pair_row.update(
                summary
            )

            pair_rows.append(
                pair_row
            )

            for fold_result in result[
                "fold_results"
            ]:

                fold_row = {
                    "pair_index": int(
                        row_i
                    ),
                    "value_a": value_a,
                    "value_b": value_b,
                }

                fold_row.update(
                    fold_result
                )

                fold_rows.append(
                    fold_row
                )

        except Exception as error:

            pair_rows.append(
                {
                    "pair_index": int(
                        row_i
                    ),
                    "value_a": value_a,
                    "value_b": value_b,
                    "status": "failed",
                    "error": str(
                        error
                    ),
                }
            )

    return (
        pd.DataFrame(
            pair_rows
        ),
        pd.DataFrame(
            fold_rows
        ),
    )


# =============================================================================
# PARALLEL SINGLE-SUBJECT WORKER
# =============================================================================


def _decode_single_subject_pair_worker(
    pair_entry,
    pair_count,
    pair_columns,
    X,
    classifier,
    scaler,
    C,
    max_iter,
    class_weight,
    random_state,
    metrics,
    verbose,
    pca_components,
):
    """
    Decode one pair.

    The pair plan already contains all CV observation indices, avoiding
    repeated np.isin computations at each temporal/frequency point.
    """

    row_i = pair_entry[
        "pair_index"
    ]

    row = pair_entry[
        "row"
    ]

    value_a = pair_entry[
        "value_a"
    ]

    value_b = pair_entry[
        "value_b"
    ]

    if verbose:

        print(
            f"[PAIR {row_i + 1}/"
            f"{pair_count}] "
            f"{value_a} vs {value_b}",
            flush=True,
        )

    try:

        result = (
            decode_pair_single_subject_cv_from_plan(
                X=X,
                pair_entry=pair_entry,
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                metrics=metrics,
                pca_components=pca_components,
            )
        )

        summary = result[
            "summary"
        ]

        pair_row = {
            "pair_index": int(
                row_i
            ),
            "value_a": value_a,
            "value_b": value_b,
            "status": "ok",
        }

        for col in pair_columns:

            if col not in pair_row:

                pair_row[
                    col
                ] = row[
                    col
                ]

        pair_row.update(
            summary
        )

        fold_rows = []

        for fold_result in result[
            "fold_results"
        ]:

            fold_row = {
                "pair_index": int(
                    row_i
                ),
                "value_a": value_a,
                "value_b": value_b,
            }

            fold_row.update(
                fold_result
            )

            fold_rows.append(
                fold_row
            )

        return (
            pair_row,
            fold_rows,
        )

    except Exception as error:

        return (
            {
                "pair_index": int(
                    row_i
                ),
                "value_a": value_a,
                "value_b": value_b,
                "status": "failed",
                "error": str(
                    error
                ),
            },
            [],
        )


# =============================================================================
# MAIN PAIRWISE SINGLE-SUBJECT ENTRY
# =============================================================================


def run_pairwise_single_subject_cv_decoding(
    X,
    values,
    pair_table,
    splits,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    verbose=True,
    n_jobs=1,
    pair_plan=None,
    pca_components=None,
):
    """
    Run pairwise decoding for one subject/session.

    Parallelism is across identity pairs.

    Notes
    -----
    `batch_size="auto"` allows joblib to aggregate very small pair jobs,
    reducing some scheduling overhead compared with forcing a separate dispatch
    for every individual pair.

    PCA, if enabled, is fitted independently inside every CV training fold.
    """

    required_cols = {
        "value_a",
        "value_b",
    }

    if not required_cols.issubset(
        pair_table.columns
    ):

        raise KeyError(
            "pair_table must contain "
            f"columns: {required_cols}"
        )

    if pair_plan is None:

        pair_plan = (
            build_single_subject_pair_plan(
                values=values,
                pair_table=pair_table,
                splits=splits,
            )
        )

    if len(
        pair_plan
    ) != len(
        pair_table
    ):

        raise ValueError(
            "pair_plan length does not "
            "match pair_table length."
        )

    for (
        expected_index,
        pair_entry,
    ) in enumerate(
        pair_plan
    ):

        if (
            pair_entry[
                "pair_index"
            ]
            != expected_index
        ):

            raise ValueError(
                "pair_plan order does not match "
                "pair_table order: "
                f"expected pair_index="
                f"{expected_index}, "
                f"found pair_index="
                f"{pair_entry['pair_index']}."
            )

    pair_count = len(
        pair_table
    )

    pair_columns = list(
        pair_table.columns
    )

    # Serial mode avoids joblib overhead entirely.
    if n_jobs == 1:

        results = [
            _decode_single_subject_pair_worker(
                pair_entry=pair_entry,
                pair_count=pair_count,
                pair_columns=pair_columns,
                X=X,
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                metrics=metrics,
                verbose=verbose,
                pca_components=pca_components,
            )
            for pair_entry
            in pair_plan
        ]

    else:

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
            pre_dispatch="2*n_jobs",
        )(
            delayed(
                _decode_single_subject_pair_worker
            )(
                pair_entry=pair_entry,
                pair_count=pair_count,
                pair_columns=pair_columns,
                X=X,
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                metrics=metrics,
                verbose=verbose,
                pca_components=pca_components,
            )
            for pair_entry
            in pair_plan
        )

    pair_rows = []
    fold_rows = []

    for (
        pair_row,
        pair_fold_rows,
    ) in results:

        pair_rows.append(
            pair_row
        )

        fold_rows.extend(
            pair_fold_rows
        )

    return (
        pd.DataFrame(
            pair_rows
        ),
        pd.DataFrame(
            fold_rows
        ),
    )


# =============================================================================
# OUTPUT METADATA
# =============================================================================


def build_pairwise_result_meta(
    decoding_type,
    pair_column,
    projection_info=None,
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    metrics=None,
    pca_components=None,
):
    """
    Build metadata dictionary for pairwise decoding outputs.
    """

    return {
        "decoding_type": (
            decoding_type
        ),

        "pair_column": (
            pair_column
        ),

        "projection_info": (
            projection_info
        ),

        "classifier_config": (
            classifier_config_dict(
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                pca_components=pca_components,
            )
        ),

        "metrics": (
            metrics
        ),
    }
