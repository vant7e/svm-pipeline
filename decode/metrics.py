# svm/decoding/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
]


SUPPORTED_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "confusion_matrix",
}


def compute_classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    metrics=None,
    average="binary",
    labels=None,
    zero_division=0,
):
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    y_score : array-like or None
        Decision values or predicted probabilities.
        Required for AUC.

    metrics : list[str] or None
        Example:
            ["accuracy", "balanced_accuracy", "auc"]

    average : str
        Used for precision/recall/F1.
        Options include:
            "binary", "macro", "micro", "weighted"

    labels : list or None
        Label ordering for confusion matrix.

    Returns
    -------
    result : dict
    """

    if metrics is None:
        metrics = DEFAULT_METRICS

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {}

    for metric in metrics:

        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Options: {sorted(SUPPORTED_METRICS)}"
            )

        if metric == "accuracy":
            result["accuracy"] = float(
                accuracy_score(y_true, y_pred)
            )

        elif metric == "balanced_accuracy":
            result["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, y_pred)
            )

        elif metric == "precision":
            result["precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=zero_division,
                )
            )

        elif metric == "recall":
            result["recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=zero_division,
                )
            )

        elif metric == "f1":
            result["f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=zero_division,
                )
            )

        elif metric == "auc":
            if y_score is None:
                result["auc"] = np.nan
            else:
                result["auc"] = safe_auc_score(
                    y_true=y_true,
                    y_score=y_score,
                )

        elif metric == "confusion_matrix":
            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
            )
            result["confusion_matrix"] = cm.tolist()

    return result


def safe_auc_score(y_true, y_score):
    """
    Compute AUC safely.

    Returns np.nan if:
        - only one class is present
        - y_score shape is incompatible
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        return float("nan")

    try:
        # Binary case
        if y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))

        # If probability matrix, use positive class column
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))

        # Multiclass case
        if y_score.ndim == 2 and y_score.shape[1] > 2:
            return float(
                roc_auc_score(
                    y_true,
                    y_score,
                    multi_class="ovr",
                    average="macro",
                )
            )

    except Exception:
        return float("nan")

    return float("nan")


def compute_chance_level(y_true, strategy="majority"):
    """
    Compute empirical chance level.

    Parameters
    ----------
    strategy : str
        majority:
            chance = proportion of largest class

        uniform:
            chance = 1 / number of classes
    """

    y_true = np.asarray(y_true)
    classes, counts = np.unique(y_true, return_counts=True)

    if strategy == "majority":
        return float(counts.max() / counts.sum())

    if strategy == "uniform":
        return float(1.0 / len(classes))

    raise ValueError("strategy must be 'majority' or 'uniform'")


def compute_fold_result(
    split_id,
    y_true,
    y_pred,
    y_score=None,
    metrics=None,
    average="binary",
    labels=None,
    extra_info=None,
):
    """
    Compute metrics for one CV fold.

    Returns
    -------
    result : dict
    """

    metric_result = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        metrics=metrics,
        average=average,
        labels=labels,
    )

    result = {
        "split_id": split_id,
        "n_test": int(len(y_true)),
        "chance_majority": compute_chance_level(
            y_true,
            strategy="majority",
        ),
        "chance_uniform": compute_chance_level(
            y_true,
            strategy="uniform",
        ),
    }

    result.update(metric_result)

    if extra_info is not None:
        result.update(extra_info)

    return result


def aggregate_fold_results(
    fold_results,
    metric_names=None,
):
    """
    Aggregate fold-level metrics.

    Parameters
    ----------
    fold_results : list[dict]

    metric_names : list[str] or None
        If None, automatically aggregate numeric keys.

    Returns
    -------
    summary : dict
    """

    if len(fold_results) == 0:
        raise ValueError("fold_results is empty")

    if metric_names is None:
        metric_names = []

        for k, v in fold_results[0].items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if k not in {"split_id"}:
                    metric_names.append(k)

    summary = {
        "n_folds": len(fold_results),
    }

    for metric in metric_names:
        values = []

        for r in fold_results:
            if metric in r:
                values.append(r[metric])

        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            continue

        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
        summary[f"{metric}_sem"] = float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        summary[f"{metric}_median"] = float(np.nanmedian(values))
        summary[f"{metric}_min"] = float(np.nanmin(values))
        summary[f"{metric}_max"] = float(np.nanmax(values))

    return summary


def classification_report_dict(
    y_true,
    y_pred,
    zero_division=0,
):
    """
    Return sklearn classification report as dictionary.
    """

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=zero_division,
    )


def flatten_confusion_matrix(cm, prefix="cm"):
    """
    Flatten confusion matrix into dict.

    Useful for saving to CSV.
    """

    cm = np.asarray(cm)

    out = {}

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            out[f"{prefix}_{i}_{j}"] = int(cm[i, j])

    return out


def merge_metric_dicts(dicts):
    """
    Merge list of metric dicts into a table-like dict of lists.

    Useful before converting to pandas DataFrame.
    """

    keys = sorted(set().union(*[d.keys() for d in dicts]))

    merged = {k: [] for k in keys}

    for d in dicts:
        for k in keys:
            merged[k].append(d.get(k, np.nan))

    return merged


def print_metric_summary(summary):
    """
    Print readable aggregated metric summary.
    """

    print("\n==============================")
    print("Metric Summary")
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v}")

    print("==============================\n")