# svm/decoding/classifiers.py

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


VALID_CLASSIFIERS = {
    "linear_svm",
    "rbf_svm",
    "logistic",
    "ridge",
    "sgd_hinge",
    "sgd_log",
    "lda",
    "gaussian_nb",
    "random_forest",
}


VALID_SCALERS = {
    "standard",
    "robust",
    "minmax",
    "none",
}


def build_scaler(scaler="standard"):
    """
    Build feature scaler.

    Options
    -------
    standard:
        StandardScaler. Recommended default for SVM/logistic/ridge.

    robust:
        RobustScaler. Useful when features contain outliers.

    minmax:
        MinMaxScaler.

    none:
        No scaling.
    """

    if scaler not in VALID_SCALERS:
        raise ValueError(
            f"Invalid scaler: {scaler}. Options: {sorted(VALID_SCALERS)}"
        )

    if scaler == "standard":
        return StandardScaler()

    if scaler == "robust":
        return RobustScaler()

    if scaler == "minmax":
        return MinMaxScaler()

    if scaler == "none":
        return None

    raise RuntimeError(f"Unhandled scaler: {scaler}")


def build_classifier(
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    probability=False,
    n_estimators=200,
    n_jobs=None,
):
    """
    Build a classifier pipeline.

    Parameters
    ----------
    classifier : str
        Options:
            linear_svm
            rbf_svm
            logistic
            ridge
            sgd_hinge
            sgd_log
            lda
            gaussian_nb
            random_forest

    scaler : str
        standard / robust / minmax / none

    C : float
        Regularization strength for SVM/logistic.

    max_iter : int
        Maximum iterations.

    class_weight : dict, "balanced", or None
        Useful for imbalanced labels.

    probability : bool
        Only used for rbf_svm. Enables predict_proba but slows training.

    n_estimators : int
        Used for random_forest.

    n_jobs : int or None
        Used where supported.
    """

    if classifier not in VALID_CLASSIFIERS:
        raise ValueError(
            f"Invalid classifier: {classifier}. "
            f"Options: {sorted(VALID_CLASSIFIERS)}"
        )

    scaler_obj = build_scaler(scaler)

    if classifier == "linear_svm":
        clf = LinearSVC(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif classifier == "rbf_svm":
        clf = SVC(
            C=C,
            kernel="rbf",
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
        )

    elif classifier == "logistic":
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif classifier == "ridge":
        clf = RidgeClassifier(
            alpha=1.0 / C,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif classifier == "sgd_hinge":
        clf = SGDClassifier(
            loss="hinge",
            alpha=1.0 / C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif classifier == "sgd_log":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1.0 / C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif classifier == "lda":
        clf = LinearDiscriminantAnalysis()

    elif classifier == "gaussian_nb":
        clf = GaussianNB()

    elif classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    else:
        raise RuntimeError(f"Unhandled classifier: {classifier}")

    if scaler_obj is None:
        return clf

    return make_pipeline(scaler_obj, clf)


def fit_classifier(clf, X_train, y_train):
    """
    Fit classifier.
    """
    clf.fit(X_train, y_train)
    return clf


def predict_classifier(clf, X_test):
    """
    Predict class labels.
    """
    return clf.predict(X_test)


def decision_values(clf, X_test):
    """
    Return decision values if available.

    Works for:
        - LinearSVC
        - SVC
        - LogisticRegression
        - RidgeClassifier
        - SGDClassifier
    """

    if hasattr(clf, "decision_function"):
        return clf.decision_function(X_test)

    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)

    # sklearn Pipeline case
    if hasattr(clf, "named_steps"):
        final_step = list(clf.named_steps.values())[-1]

        if hasattr(clf, "decision_function"):
            return clf.decision_function(X_test)

        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_test)

        if hasattr(final_step, "decision_function"):
            return clf.decision_function(X_test)

        if hasattr(final_step, "predict_proba"):
            return clf.predict_proba(X_test)

    return None


def classifier_config_dict(
    classifier="linear_svm",
    scaler="standard",
    C=1.0,
    max_iter=10000,
    class_weight=None,
    random_state=42,
    probability=False,
    n_estimators=200,
    n_jobs=None,
):
    """
    Store classifier settings in output metadata.
    """

    return {
        "classifier": classifier,
        "scaler": scaler,
        "C": C,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "random_state": random_state,
        "probability": probability,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
    }


def recommended_classifier_notes():
    """
    Human-readable recommendations.
    """

    return {
        "linear_svm": (
            "Recommended default for high-dimensional MEG/EEG decoding. "
            "Fast, interpretable, works well when features >> samples."
        ),
        "logistic": (
            "Useful alternative to linear SVM. Provides probabilistic-style "
            "decision values and is often stable for binary decoding."
        ),
        "ridge": (
            "Good linear baseline for high-dimensional data. Often very fast."
        ),
        "sgd_hinge": (
            "Scalable linear SVM-like classifier for very large feature spaces."
        ),
        "sgd_log": (
            "Scalable logistic-style classifier for very large feature spaces."
        ),
        "lda": (
            "Classic decoding classifier, but may be unstable when features "
            "greatly exceed samples unless dimensionality is reduced."
        ),
        "gaussian_nb": (
            "Simple baseline. Usually not ideal for correlated MEG/EEG features."
        ),
        "rbf_svm": (
            "Nonlinear SVM. Usually too slow for large MEG/EEG feature spaces."
        ),
        "random_forest": (
            "Nonlinear baseline. Can be useful, but often less ideal for "
            "very high-dimensional neural time-series features."
        ),
    }


def print_available_classifiers():
    """
    Print available classifier options.
    """

    notes = recommended_classifier_notes()

    print("\nAvailable classifiers")
    print("---------------------")

    for name in sorted(VALID_CLASSIFIERS):
        print(f"{name}: {notes.get(name, '')}")

    print("")