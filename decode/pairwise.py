# svm/decode/pairwise.py

import numpy as np
import pandas as pd

from svm.decode.classifiers import (
    build_classifier,
    fit_classifier,
    predict_classifier,
    decision_values,
    classifier_config_dict,
)

from svm.decode.metrics import (
    compute_fold_result,
    aggregate_fold_results,
)

from svm.decode.splits import (
    make_loso_splits,
)


def get_pair_trial_indices(values, value_a, value_b):
    """
    Get trial indices for two selected classes/items.

    Parameters
    ----------
    values : array-like
        One label value per trial.

    value_a, value_b :
        The two classes/items to decode.

    Returns
    -------
    idx : np.ndarray
        Trial indices containing value_a or value_b.

    y : np.ndarray
        Binary labels:
            value_a -> 0
            value_b -> 1
    """

    values = np.asarray(values)

    mask_a = values == value_a
    mask_b = values == value_b

    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]

    if len(idx_a) == 0:
        raise ValueError(f"No trials found for value_a={value_a}")

    if len(idx_b) == 0:
        raise ValueError(f"No trials found for value_b={value_b}")

    idx = np.concatenate([idx_a, idx_b])
    y = np.concatenate([
        np.zeros(len(idx_a), dtype=int),
        np.ones(len(idx_b), dtype=int),
    ])

    return idx, y


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
):
    """
    Decode one pair within one subject using trial-level CV.

    This works for:
        - leave-one-run-out
        - leave-one-block-out
        - leave-one-observation-out
        - stratified k-fold

    Parameters
    ----------
    X : np.ndarray
        Shape:
            (trial, features)

    values : array-like
        One decoding label per trial.

    value_a, value_b :
        Pair values.

    splits : list[dict]
        Trial-level splits with train_idx and test_idx.

    Returns
    -------
    result : dict
    """

    values = np.asarray(values)

    pair_idx, pair_y = get_pair_trial_indices(
        values=values,
        value_a=value_a,
        value_b=value_b,
    )

    fold_results = []

    for split in splits:

        train_idx_global = np.asarray(split["train_idx"])
        test_idx_global = np.asarray(split["test_idx"])

        # keep only pair trials inside each split
        train_mask = np.isin(pair_idx, train_idx_global)
        test_mask = np.isin(pair_idx, test_idx_global)

        train_pair_idx = pair_idx[train_mask]
        test_pair_idx = pair_idx[test_mask]

        y_train = pair_y[train_mask]
        y_test = pair_y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        if len(np.unique(y_test)) < 2:
            continue

        X_train = X[train_pair_idx]
        X_test = X[test_pair_idx]

        clf = build_classifier(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

        clf = fit_classifier(clf, X_train, y_train)
        y_pred = predict_classifier(clf, X_test)
        y_score = decision_values(clf, X_test)

        fold_result = compute_fold_result(
            split_id=split["split_id"],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            average="binary",
            extra_info={
                "value_a": value_a,
                "value_b": value_b,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "split_type": split.get("split_type"),
            },
        )

        fold_results.append(fold_result)

    if len(fold_results) == 0:
        raise ValueError(
            f"No valid folds for pair {value_a} vs {value_b}."
        )

    summary = aggregate_fold_results(fold_results)

    return {
        "value_a": value_a,
        "value_b": value_b,
        "fold_results": fold_results,
        "summary": summary,
    }


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
):
    """
    Decode one pair using leave-one-subject-out.

    Intended for datasets where each subject has one trial/image matrix:

        X_list = [X_subject_1, X_subject_2, ...]

    and the same pair values exist across subjects.

    Parameters
    ----------
    X_list : list[np.ndarray]
        Each X has shape:
            (trial, features)

    values_list : list[array-like]
        One label vector per subject.

    value_a, value_b :
        Pair values.

    Returns
    -------
    result : dict
    """

    if len(X_list) != len(values_list):
        raise ValueError(
            "X_list and values_list must have the same length."
        )

    n_subjects = len(X_list)
    splits = make_loso_splits(n_subjects)

    fold_results = []

    for split in splits:

        train_subjects = split["train_subjects"]
        test_subject = split["test_subjects"][0]

        X_train_all = []
        y_train_all = []

        for si in train_subjects:
            idx, y = get_pair_trial_indices(
                values=values_list[si],
                value_a=value_a,
                value_b=value_b,
            )

            X_train_all.append(X_list[si][idx])
            y_train_all.append(y)

        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)

        test_idx, y_test = get_pair_trial_indices(
            values=values_list[test_subject],
            value_a=value_a,
            value_b=value_b,
        )

        X_test = X_list[test_subject][test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        if len(np.unique(y_test)) < 2:
            continue

        clf = build_classifier(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

        clf = fit_classifier(clf, X_train, y_train)
        y_pred = predict_classifier(clf, X_test)
        y_score = decision_values(clf, X_test)

        fold_result = compute_fold_result(
            split_id=split["split_id"],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            average="binary",
            extra_info={
                "value_a": value_a,
                "value_b": value_b,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "split_type": "loso",
                "test_subject": int(test_subject),
            },
        )

        fold_results.append(fold_result)

    if len(fold_results) == 0:
        raise ValueError(
            f"No valid LOSO folds for pair {value_a} vs {value_b}."
        )

    summary = aggregate_fold_results(fold_results)

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
):
    """
    Run pairwise decoding across many pairs using LOSO.

    Parameters
    ----------
    X_list : list[np.ndarray]
        One classifier matrix per subject.

    values_list : list[array-like]
        One metadata label vector per subject.
        Example:
            image labels
            category labels
            confidence labels

    pair_table : pd.DataFrame
        Must contain:
            value_a
            value_b

    Returns
    -------
    pair_results_df : pd.DataFrame
        One row per pair.

    fold_results_df : pd.DataFrame
        One row per pair per fold.
    """

    required_cols = {"value_a", "value_b"}

    if not required_cols.issubset(pair_table.columns):
        raise KeyError(
            f"pair_table must contain columns: {required_cols}"
        )

    pair_rows = []
    fold_rows = []

    for row_i, row in pair_table.iterrows():

        value_a = row["value_a"]
        value_b = row["value_b"]

        if verbose:
            print(
                f"[PAIR {row_i + 1}/{len(pair_table)}] "
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
            )

            summary = result["summary"]

            pair_row = {
                "pair_index": int(row_i),
                "value_a": value_a,
                "value_b": value_b,
                "status": "ok",
            }

            for col in pair_table.columns:
                if col not in pair_row:
                    pair_row[col] = row[col]

            pair_row.update(summary)
            pair_rows.append(pair_row)

            for fold_result in result["fold_results"]:
                fold_row = {
                    "pair_index": int(row_i),
                    "value_a": value_a,
                    "value_b": value_b,
                }
                fold_row.update(fold_result)
                fold_rows.append(fold_row)

        except Exception as e:
            pair_rows.append({
                "pair_index": int(row_i),
                "value_a": value_a,
                "value_b": value_b,
                "status": "failed",
                "error": str(e),
            })

    pair_results_df = pd.DataFrame(pair_rows)
    fold_results_df = pd.DataFrame(fold_rows)

    return pair_results_df, fold_results_df


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
):
    """
    Run pairwise decoding for one subject/session using trial-level CV.

    This is useful for:
        - leave-one-run-out
        - leave-one-block-out
        - leave-one-observation-out
        - stratified k-fold
    """

    required_cols = {"value_a", "value_b"}

    if not required_cols.issubset(pair_table.columns):
        raise KeyError(
            f"pair_table must contain columns: {required_cols}"
        )

    pair_rows = []
    fold_rows = []

    for row_i, row in pair_table.iterrows():

        value_a = row["value_a"]
        value_b = row["value_b"]

        if verbose:
            print(
                f"[PAIR {row_i + 1}/{len(pair_table)}] "
                f"{value_a} vs {value_b}",
                flush=True,
            )

        try:
            result = decode_pair_single_subject_cv(
                X=X,
                values=values,
                value_a=value_a,
                value_b=value_b,
                splits=splits,
                classifier=classifier,
                scaler=scaler,
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                metrics=metrics,
            )

            summary = result["summary"]

            pair_row = {
                "pair_index": int(row_i),
                "value_a": value_a,
                "value_b": value_b,
                "status": "ok",
            }

            for col in pair_table.columns:
                if col not in pair_row:
                    pair_row[col] = row[col]

            pair_row.update(summary)
            pair_rows.append(pair_row)

            for fold_result in result["fold_results"]:
                fold_row = {
                    "pair_index": int(row_i),
                    "value_a": value_a,
                    "value_b": value_b,
                }
                fold_row.update(fold_result)
                fold_rows.append(fold_row)

        except Exception as e:
            pair_rows.append({
                "pair_index": int(row_i),
                "value_a": value_a,
                "value_b": value_b,
                "status": "failed",
                "error": str(e),
            })

    pair_results_df = pd.DataFrame(pair_rows)
    fold_results_df = pd.DataFrame(fold_rows)

    return pair_results_df, fold_results_df


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
):
    """
    Build metadata dictionary for pairwise decoding outputs.
    """

    return {
        "decoding_type": decoding_type,
        "pair_column": pair_column,
        "projection_info": projection_info,
        "classifier_config": classifier_config_dict(
            classifier=classifier,
            scaler=scaler,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "metrics": metrics,
    }