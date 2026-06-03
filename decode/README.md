# `svm.decode`

`svm.decode` contains the decoding layer of the general neural decoding framework.

This package converts projected neural features into classifier results.  
It is designed to work with standardized feature matrices produced by:

```text
svm.features
svm.core.time
svm.decode.projection
```

The main classifier input format is:

```text
(trial, features)
```

This package supports:

```text
classifier construction
feature projection
cross-validation split construction
pairwise decoding
metric computation
fold-level result aggregation
```

---

# Modules

## `classifiers.py`

Builds classifier pipelines for neural decoding.

Supported classifiers:

```text
linear_svm
rbf_svm
logistic
ridge
sgd_hinge
sgd_log
lda
gaussian_nb
random_forest
```

Supported scalers:

```text
standard
robust
minmax
none
```

Main utilities:

```python
build_scaler()

build_classifier()

fit_classifier()

predict_classifier()

decision_values()

classifier_config_dict()

recommended_classifier_notes()

print_available_classifiers()
```

Recommended default:

```text
linear_svm + standard scaler
```

This is usually appropriate for high-dimensional MEG/EEG features where:

```text
features >> samples
```

Other useful options:

```text
logistic:
    linear probabilistic-style alternative

ridge:
    fast linear baseline

sgd_hinge:
    scalable SVM-like classifier for very large feature spaces

sgd_log:
    scalable logistic-style classifier

lda:
    classic decoding model, but less stable when features greatly exceed samples

gaussian_nb:
    simple baseline

rbf_svm:
    nonlinear SVM, usually slow for large neural feature spaces

random_forest:
    nonlinear baseline, often less ideal for high-dimensional time-series features
```

---

## `metrics.py`

Computes classification metrics and aggregates fold-level results.

Default metrics:

```text
accuracy
balanced_accuracy
```

Supported metrics:

```text
accuracy
balanced_accuracy
precision
recall
f1
auc
confusion_matrix
```

Main utilities:

```python
compute_classification_metrics()

safe_auc_score()

compute_chance_level()

compute_fold_result()

aggregate_fold_results()

classification_report_dict()

flatten_confusion_matrix()

merge_metric_dicts()

print_metric_summary()
```

Fold-level outputs include:

```text
split_id
n_test
chance_majority
chance_uniform
accuracy
balanced_accuracy
precision
recall
f1
auc
confusion_matrix
```

Aggregated outputs include mean, standard deviation, SEM, median, min, and max for numeric fold metrics.

---

## `pairwise.py`

Runs pairwise decoding between two label values.

This module supports two major decoding modes:

```text
1. Leave-one-subject-out decoding
2. Single-subject / trial-level cross-validation decoding
```

Main utilities:

```python
get_pair_trial_indices()

decode_pair_single_subject_cv()

decode_pair_loso()

run_pairwise_loso_decoding()

run_pairwise_single_subject_cv_decoding()

build_pairwise_result_meta()
```

Pairwise decoding assumes a pair table with:

```text
value_a
value_b
```

For each pair:

```text
value_a -> class 0
value_b -> class 1
```

Example pair columns can come from any metadata field:

```text
image
category
OLD_NEW
Confidence
stimulus_identity
semantic_label
```

### Leave-one-subject-out decoding

Used when there are multiple subject-level feature matrices:

```text
X_list = [X_subject_1, X_subject_2, ...]
values_list = [labels_subject_1, labels_subject_2, ...]
```

Each fold trains on all subjects except one and tests on the held-out subject.

### Single-subject CV decoding

Used when one subject/session has trial-level splits:

```text
X = (trial, features)
values = label per trial
splits = train/test trial indices
```

This supports designs such as:

```text
leave-one-run-out
leave-one-block-out
leave-one-observation-out
stratified k-fold
```

---

## `projection.py`

Projects standardized feature tensors into classifier-ready matrices.

Input feature format:

```text
(freq, time, trial, channel_or_source)
```

Output format:

```text
(trial, features)
```

Supported projection modes:

```text
point
window
full
mean
sliding
```

Main utilities:

```python
project_feature_to_X()

project_multiple_subjects_to_X()

project_multiple_subjects_sliding_to_X()

subset_X_by_trials()

check_X_list_alignment()

summarize_projection_info()

print_projection_summary()
```

### Time projection modes

```text
point:
    select one nearest time point

window:
    select a time window and flatten frequency × time × channel

full:
    use the full available time axis and flatten

mean:
    average across time window before flattening

sliding:
    generate multiple sliding-window X matrices
```

### Spatial selection

Spatial selection is delegated to `svm.core.spatial`.

Supported spatial methods include:

```text
all
prefix
name
regex
index
```

Named spatial selections can also be used through ROI maps such as:

```text
occipital
temporal
occipito_temporal
posterior
frontal
```

This module does not select labels.  
It only converts neural features into classifier-ready matrices.

---

## `splits.py`

Builds cross-validation split structures.

Supported split types:

```text
leave-one-subject-out
leave-one-group-out
leave-one-run-out
leave-one-block-out
leave-one-observation-out
stratified k-fold
```

Main utilities:

```python
make_loso_splits()

make_leave_one_group_out_splits()

make_leave_one_run_out_splits()

make_leave_one_block_out_splits()

make_leave_one_observation_out_splits()

make_stratified_kfold_splits()

validate_split_indices()

apply_trial_split()

describe_splits()

print_split_summary()
```

### Subject-level splits

For leave-one-subject-out decoding:

```text
train_subjects = all subjects except one
test_subjects = held-out subject
```

### Trial-level splits

For trial-level decoding, splits contain:

```text
train_idx
test_idx
```

These can be used for:

```text
leave-one-run-out
leave-one-block-out
leave-one-observation-out
stratified k-fold
```

---

# Standard Input and Output

## Input to decoding

Most decoding functions expect classifier matrices:

```text
X = (trial, features)
```

For multi-subject LOSO decoding:

```text
X_list = [
    X_subject_1,
    X_subject_2,
    ...
]
```

For labels:

```text
values = one label per trial
values_list = one label vector per subject
```

---

## Output from pairwise decoding

Pairwise decoding returns:

```text
pair_results_df
fold_results_df
```

### `pair_results_df`

One row per decoded pair.

Typical columns:

```text
pair_index
value_a
value_b
status
accuracy_mean
balanced_accuracy_mean
accuracy_std
accuracy_sem
n_folds
```

### `fold_results_df`

One row per pair per fold.

Typical columns:

```text
pair_index
value_a
value_b
split_id
split_type
n_train
n_test
accuracy
balanced_accuracy
chance_majority
chance_uniform
```

These outputs can later be merged, converted to matrices, and transformed into RDMs.

---

# Framework Role

Within the full pipeline:

```text
svm.features
    ->
standardized feature package

svm.core.time / svm.core.spatial
    ->
projection utilities

svm.decode.projection
    ->
X = (trial, features)

svm.decode.splits
    ->
train/test structure

svm.decode.classifiers
    ->
classifier pipeline

svm.decode.pairwise
    ->
pairwise decoding results

svm.decode.metrics
    ->
fold-level and aggregated metrics
```

---

# Design Principle

`svm.decode` should remain task-general.

It should not hard-code:

```text
subject IDs
specific datasets
stimulus names
metadata column names
ROI names
experimental conditions
```

All dataset-specific information should be passed through:

```text
manifest files
metadata labels
CLI arguments
configuration files
```

---

# Current Supported Decoding Workflows

The current decoding layer supports:

```text
single-window pairwise decoding
full-window pairwise decoding
timepoint pairwise decoding
sliding-window projection
leave-one-subject-out decoding
single-subject trial-level CV decoding
multiple classifier backends
multiple metrics
matrix/RDM-ready outputs
```

---

# Example Usage

Example static pairwise decoding:

```bash
python -m svm.scripts.run_pairwise_decoding \
  --manifest /path/to/manifest.csv \
  --out_dir /path/to/results \
  --pair_column image \
  --pair_mode sample \
  --n_pairs 50 \
  --time_mode full \
  --tmin 0.0 \
  --tmax 0.6 \
  --spatial_preset occipito_temporal \
  --classifier linear_svm \
  --scaler standard
```

Example single-timepoint projection:

```bash
python -m svm.scripts.run_pairwise_decoding \
  --manifest /path/to/manifest.csv \
  --out_dir /path/to/results \
  --pair_column image \
  --pair_mode sample \
  --n_pairs 50 \
  --time_mode point \
  --target_time 0.150 \
  --spatial_preset occipito_temporal \
  --classifier linear_svm
```

---

# Notes

For high-dimensional MEG/EEG decoding, the recommended baseline is:

```text
linear_svm + standard scaler
```

For very large feature spaces, consider:

```text
sgd_hinge
sgd_log
ridge
```

For small sample sizes with many features, nonlinear classifiers such as RBF SVM or random forest may be slow or unstable.
