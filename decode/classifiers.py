# svm/decode/classifiers.py

from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
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


# =============================================================================
# SCALER
# =============================================================================


def build_scaler(
    scaler: str = "standard",
):
    """
    Build feature scaler.

    Scaling is applied inside the sklearn Pipeline, therefore all scaler
    parameters are estimated from the training fold only.

    Options
    -------
    standard
        StandardScaler.

    robust
        RobustScaler.

    minmax
        MinMaxScaler.

    none
        No scaling.
    """

    if scaler not in VALID_SCALERS:
        raise ValueError(
            f"Invalid scaler: {scaler}. "
            f"Options: {sorted(VALID_SCALERS)}"
        )

    if scaler == "standard":
        return StandardScaler()

    if scaler == "robust":
        return RobustScaler()

    if scaler == "minmax":
        return MinMaxScaler()

    if scaler == "none":
        return None

    raise RuntimeError(
        f"Unhandled scaler: {scaler}"
    )


# =============================================================================
# PCA
# =============================================================================


def validate_pca_components(
    pca_components: float | int | None,
) -> float | int | None:
    """
    Validate PCA specification.

    None
        PCA disabled.

    float in (0, 1)
        Retain the requested fraction of explained variance.
        Example:
            0.95 -> retain 95% of variance.

    positive integer
        Keep exactly that many principal components.
    """

    if pca_components is None:
        return None

    if isinstance(
        pca_components,
        bool,
    ):
        raise TypeError(
            "pca_components cannot be boolean."
        )

    if isinstance(
        pca_components,
        int,
    ):
        if pca_components <= 0:
            raise ValueError(
                "Integer pca_components must be positive."
            )

        return pca_components

    value = float(
        pca_components
    )

    if 0.0 < value < 1.0:
        return value

    if value >= 1.0 and value.is_integer():
        integer = int(
            value
        )

        if integer <= 0:
            raise ValueError(
                "Integer pca_components must be positive."
            )

        return integer

    raise ValueError(
        "pca_components must be None, "
        "a variance fraction in (0, 1), "
        "or a positive integer."
    )


def build_pca(
    pca_components: float | int | None,
    *,
    random_state: int = 42,
):
    """
    Construct optional PCA transformer.

    PCA is intentionally placed inside the sklearn Pipeline so fitting occurs
    on X_train only within every CV fold.
    """

    pca_components = validate_pca_components(
        pca_components
    )

    if pca_components is None:
        return None

    # For variance-fraction PCA (e.g., 0.95), full SVD is deterministic and
    # supports explained-variance selection directly.
    #
    # For integer PCA, "auto" allows sklearn to choose an efficient solver.
    if isinstance(
        pca_components,
        float,
    ):
        svd_solver = "full"

    else:
        svd_solver = "auto"

    return PCA(
        n_components=pca_components,
        svd_solver=svd_solver,
        random_state=random_state,
    )


# =============================================================================
# CLASSIFIER
# =============================================================================


def build_classifier(
    classifier: str = "linear_svm",
    scaler: str = "standard",
    C: float = 1.0,
    max_iter: int = 10000,
    class_weight=None,
    random_state: int = 42,
    probability: bool = False,
    n_estimators: int = 200,
    n_jobs: int | None = None,
    pca_components: float | int | None = None,
):
    """
    Build classifier preprocessing + estimator Pipeline.

    Preprocessing order
    -------------------
    When both scaling and PCA are enabled:

        X_train
            -> StandardScaler.fit_transform(X_train)
            -> PCA.fit_transform(X_train)
            -> classifier.fit(...)

        X_test
            -> training scaler.transform(X_test)
            -> training PCA.transform(X_test)
            -> classifier.predict(...)

    Therefore neither scaling nor PCA sees held-out observations.

    Parameters
    ----------
    classifier
        Classifier name.

    scaler
        standard / robust / minmax / none.

    pca_components
        None:
            PCA disabled.

        float in (0, 1):
            Explained-variance threshold.
            Example: 0.95.

        positive integer:
            Fixed number of components.

    C
        Regularization strength for SVM/logistic.

    max_iter
        Maximum optimization iterations.

    class_weight
        sklearn class-weight specification.

    random_state
        Random seed.
    """

    if classifier not in VALID_CLASSIFIERS:
        raise ValueError(
            f"Invalid classifier: {classifier}. "
            f"Options: {sorted(VALID_CLASSIFIERS)}"
        )

    scaler_obj = build_scaler(
        scaler
    )

    pca_obj = build_pca(
        pca_components,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # Final estimator
    # ------------------------------------------------------------------

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

        raise RuntimeError(
            f"Unhandled classifier: {classifier}"
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    steps: list[
        tuple[
            str,
            Any,
        ]
    ] = []

    if scaler_obj is not None:

        steps.append(
            (
                "scaler",
                scaler_obj,
            )
        )

    if pca_obj is not None:

        steps.append(
            (
                "pca",
                pca_obj,
            )
        )

    steps.append(
        (
            "classifier",
            clf,
        )
    )

    # Always return Pipeline.
    #
    # This keeps downstream behavior consistent even when scaling/PCA are
    # disabled.
    return Pipeline(
        steps
    )


# =============================================================================
# FIT / PREDICT
# =============================================================================


def fit_classifier(
    clf,
    X_train,
    y_train,
):
    """Fit classifier Pipeline."""

    clf.fit(
        X_train,
        y_train,
    )

    return clf


def predict_classifier(
    clf,
    X_test,
):
    """Predict class labels."""

    return clf.predict(
        X_test
    )


def decision_values(
    clf,
    X_test,
):
    """
    Return classifier decision values when available.

    sklearn Pipeline transparently applies preprocessing before calling the
    final estimator's decision_function / predict_proba.
    """

    if hasattr(
        clf,
        "decision_function",
    ):
        return clf.decision_function(
            X_test
        )

    if hasattr(
        clf,
        "predict_proba",
    ):
        return clf.predict_proba(
            X_test
        )

    return None


# =============================================================================
# METADATA
# =============================================================================


def classifier_config_dict(
    classifier: str = "linear_svm",
    scaler: str = "standard",
    C: float = 1.0,
    max_iter: int = 10000,
    class_weight=None,
    random_state: int = 42,
    probability: bool = False,
    n_estimators: int = 200,
    n_jobs: int | None = None,
    pca_components: float | int | None = None,
):
    """
    Store complete classifier configuration in output metadata.
    """

    pca_components = validate_pca_components(
        pca_components
    )

    return {
        "classifier": classifier,
        "scaler": scaler,
        "pca_components": pca_components,
        "pca_enabled": (
            pca_components is not None
        ),
        "preprocessing_order": (
            [
                "scaler",
                "pca",
                "classifier",
            ]
            if pca_components is not None
            else [
                "scaler",
                "classifier",
            ]
        ),
        "preprocessing_fit_scope": (
            "training fold only"
        ),
        "C": C,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "random_state": random_state,
        "probability": probability,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
    }


# =============================================================================
# NOTES
# =============================================================================


def recommended_classifier_notes():
    """Human-readable classifier notes."""

    return {
        "linear_svm": (
            "Recommended default for high-dimensional EEG/MEG decoding."
        ),

        "logistic": (
            "Useful linear alternative to SVM."
        ),

        "ridge": (
            "Fast linear baseline."
        ),

        "sgd_hinge": (
            "Scalable linear SVM-like classifier."
        ),

        "sgd_log": (
            "Scalable logistic-style classifier."
        ),

        "lda": (
            "Classic decoder; PCA may help when features exceed samples."
        ),

        "gaussian_nb": (
            "Simple baseline."
        ),

        "rbf_svm": (
            "Nonlinear SVM; generally much slower."
        ),

        "random_forest": (
            "Nonlinear baseline."
        ),
    }


def print_available_classifiers():
    """Print available classifier options."""

    notes = recommended_classifier_notes()

    print(
        "\nAvailable classifiers"
    )

    print(
        "---------------------"
    )

    for name in sorted(
        VALID_CLASSIFIERS
    ):

        print(
            f"{name}: "
            f"{notes.get(name, '')}"
        )

    print()
