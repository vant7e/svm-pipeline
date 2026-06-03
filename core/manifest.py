# svm/core/manifest.py

from pathlib import Path
import pandas as pd

from svm.core.io import load_feature_meta


REQUIRED_MANIFEST_COLUMNS = {
    "subject",
    "feature_path",
    "meta_path",
}


OPTIONAL_MANIFEST_COLUMNS = {
    "task",
    "session",
    "run",
    "group",
    "feature_space",
    "representation",
    "notes",
}


def load_manifest(path):
    """
    Load dataset manifest CSV.

    Required columns:
        subject
        feature_path
        meta_path

    Optional columns:
        task
        session
        run
        group
        feature_space
        representation
        notes
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    manifest = pd.read_csv(path)

    validate_manifest(manifest)

    return manifest


def validate_manifest(manifest):
    """
    Validate manifest table.
    """

    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)

    if missing:
        raise KeyError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    if len(manifest) == 0:
        raise ValueError("Manifest is empty.")

    for col in ["feature_path", "meta_path"]:
        missing_paths = []

        for p in manifest[col].tolist():
            if not Path(p).exists():
                missing_paths.append(p)

        if missing_paths:
            raise FileNotFoundError(
                f"Some files in column '{col}' do not exist. "
                f"First missing files: {missing_paths[:5]}"
            )

    return True


def build_manifest_from_feature_dir(
    feature_dir,
    pattern="*.npy",
    output_path=None,
):
    """
    Build manifest by scanning a feature directory.

    This assumes each feature file has a matching:
        .meta.json

    Example feature:
        10665_recognition_raw_amplitude.npy

    Matching meta:
        10665_recognition_raw_amplitude.meta.json
    """

    feature_dir = Path(feature_dir)

    if not feature_dir.exists():
        raise FileNotFoundError(feature_dir)

    rows = []

    for feature_path in sorted(feature_dir.glob(pattern)):

        if feature_path.name.endswith(".meta.npy"):
            continue

        if feature_path.name.endswith(".meta.json"):
            continue

        meta_path = feature_path.with_suffix(".meta.json")

        if not meta_path.exists():
            print(f"[WARN] Missing meta for {feature_path.name}")
            continue

        meta = load_feature_meta(meta_path)

        rows.append({
            "subject": meta.get("subject"),
            "task": meta.get("task"),
            "feature_space": meta.get("feature_space"),
            "representation": meta.get("representation"),
            "feature_path": str(feature_path),
            "meta_path": str(meta_path),
            "source_epoch_file": meta.get("source_epoch_file"),
        })

    manifest = pd.DataFrame(rows)

    validate_manifest(manifest)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_path, index=False)
        print(f"[MANIFEST] Saved: {output_path}")

    return manifest


def filter_manifest(
    manifest,
    subject=None,
    task=None,
    feature_space=None,
    representation=None,
    session=None,
    group=None,
    query=None,
):
    """
    Filter manifest by common fields or pandas query.
    """

    out = manifest.copy()

    filters = {
        "subject": subject,
        "task": task,
        "feature_space": feature_space,
        "representation": representation,
        "session": session,
        "group": group,
    }

    for col, value in filters.items():

        if value is None:
            continue

        if col not in out.columns:
            raise KeyError(f"Manifest has no column: {col}")

        if isinstance(value, (list, tuple, set)):
            out = out[out[col].astype(str).isin([str(v) for v in value])]
        else:
            out = out[out[col].astype(str) == str(value)]

    if query is not None:
        out = out.query(query)

    out = out.reset_index(drop=True)

    validate_manifest(out)

    return out


def load_manifest_feature_paths(manifest):
    """
    Return feature_path and meta_path lists from manifest.
    """

    validate_manifest(manifest)

    feature_paths = manifest["feature_path"].tolist()
    meta_paths = manifest["meta_path"].tolist()

    return feature_paths, meta_paths


def get_manifest_subjects(manifest):
    """
    Return subject list from manifest.
    """

    validate_manifest(manifest)

    return manifest["subject"].astype(str).tolist()


def validate_manifest_feature_consistency(
    manifest,
    require_same_feature_space=True,
    require_same_representation=True,
    require_same_task=False,
):
    """
    Check feature metadata consistency across manifest rows.
    """

    validate_manifest(manifest)

    metas = [
        load_feature_meta(p)
        for p in manifest["meta_path"].tolist()
    ]

    if require_same_feature_space:
        spaces = {m.get("feature_space") for m in metas}
        if len(spaces) != 1:
            raise ValueError(f"Multiple feature spaces found: {spaces}")

    if require_same_representation:
        reps = {m.get("representation") for m in metas}
        if len(reps) != 1:
            raise ValueError(f"Multiple representations found: {reps}")

    if require_same_task:
        tasks = {m.get("task") for m in metas}
        if len(tasks) != 1:
            raise ValueError(f"Multiple tasks found: {tasks}")

    shapes = [m.get("shape") for m in metas]

    # allow subject-specific trial count, but require same freq/time/channel dimensions
    ref = shapes[0]
    for i, shape in enumerate(shapes[1:], start=1):
        if shape[0] != ref[0]:
            raise ValueError(f"Frequency dimension mismatch at row {i}")

        if shape[1] != ref[1]:
            raise ValueError(f"Time dimension mismatch at row {i}")

        if shape[3] != ref[3]:
            raise ValueError(f"Channel/spatial dimension mismatch at row {i}")

    return True


def save_manifest(manifest, path):
    """
    Save manifest CSV.
    """

    validate_manifest(manifest)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(path, index=False)

    print(f"[MANIFEST] Saved: {path}")


def describe_manifest(manifest):
    """
    Return readable manifest summary.
    """

    validate_manifest(manifest)

    lines = []
    lines.append("Manifest summary")
    lines.append("----------------")
    lines.append(f"Rows: {len(manifest)}")

    for col in [
        "subject",
        "task",
        "feature_space",
        "representation",
        "session",
        "group",
    ]:
        if col in manifest.columns:
            lines.append(f"{col}: {manifest[col].nunique()} unique")

    return "\n".join(lines)


def print_manifest_summary(manifest):
    """
    Print readable manifest summary.
    """

    print(describe_manifest(manifest))