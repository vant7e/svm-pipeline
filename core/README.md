# `svm.core`

`svm.core` contains the shared infrastructure for the general neural decoding framework.

These modules define the standard feature schema, file I/O, metadata/label utilities, manifest management, spatial selection, and time projection.

All feature arrays are expected to follow the same standard shape:

```text
(freq, time, trial, channel)
```

This makes the framework compatible with raw activity, oscillatory power, and phase-based representations.

---

# Modules

## `schema.py`

Defines the standard feature metadata format and validation rules.

Supports feature spaces:

```text
raw
power
phase
```

Supports representations:

```text
raw:
    amplitude

power:
    magnitude
    magnitude_squared
    log_power

phase:
    angle
    sin_cos
    complex_unit
```

Main utilities:

```python
FeatureMeta

build_feature_meta()

validate_feature_array()

validate_feature_type()

validate_feature_meta()

describe_feature()
```

Use this module whenever a new feature array is created or loaded.

---

## `io.py`

Handles saving and loading standardized feature packages.

A feature package includes:

```text
feature.npy
feature.meta.json
```

Main utilities:

```python
save_feature_array()

load_feature_array()

save_feature_meta()

load_feature_meta()

save_feature_package()

load_feature_package()

should_compute()

build_feature_filename()

build_feature_paths()

print_feature_summary()
```

This module supports both real-valued and complex-valued feature arrays.

Storage rules:

```text
real-valued:
    float32

complex-valued:
    complex64
```

---

## `labels.py`

Handles metadata and label construction.

This module is task-general.

It does not assume specific labels such as:

```text
image
OLD_NEW
Confidence
category
semantic_label
```

Main utilities:

```python
load_metadata()

validate_metadata_length()

require_columns()

get_column_values()

build_binary_labels()

build_multiclass_labels()

filter_metadata()

build_pairs_from_column()

sample_pairs()

chunk_pairs()

validate_trial_alignment()

get_trial_indices_by_value()

describe_column()
```

Typical uses:

```text
metadata column
    ->
binary labels

metadata column
    ->
multiclass labels

metadata column
    ->
pairwise decoding pairs
```

For pairwise decoding:

```python
build_pairs_from_column()
```

constructs all possible item pairs from any metadata column.

---

## `manifest.py`

Handles dataset-level feature organization.

A manifest is a CSV table listing which feature packages should be loaded together.

Required columns:

```text
subject
feature_path
meta_path
```

Optional columns may include:

```text
task
session
run
group
feature_space
representation
notes
```

Main utilities:

```python
load_manifest()

validate_manifest()

build_manifest_from_feature_dir()

filter_manifest()

load_manifest_feature_paths()

get_manifest_subjects()

validate_manifest_feature_consistency()

save_manifest()

describe_manifest()

print_manifest_summary()
```

This module allows decoding scripts to avoid hard-coded subject lists.

---

## `spatial.py`

Handles channel, sensor, and ROI selection.

Supported selection methods:

```text
all
prefix
name
regex
index
```

Main utilities:

```python
validate_ch_names()

get_spatial_indices()

apply_spatial_selection()

select_spatial_feature()

build_ctf_roi_map()

build_standard_eeg_roi_map()

get_selection_from_map()

select_by_named_space()

summarize_spatial_selection()

print_spatial_summary()
```

This supports both MEG and EEG-style spatial selection.

For CTF MEG systems, named presets may include:

```text
occipital
temporal
parietal
frontal
occipito_temporal
posterior
```

---

## `time.py`

Handles time selection and time projection.

Supported time modes:

```text
point
window
full
mean
sliding
```

Main utilities:

```python
get_time_indices()

get_nearest_time_index()

make_sliding_windows()

select_time_slice()

feature_to_classifier_matrix()

extract_classifier_features()

extract_sliding_classifier_features()

summarize_time_axis()

print_time_summary()
```

This module converts feature tensors into classifier-ready matrices:

```text
(freq, time, trial, channel)
    ->
(trial, features)
```

Supported decoding styles include:

```text
single-timepoint decoding
fixed-window decoding
full-window decoding
mean-window decoding
sliding-window decoding
```

---

# Design Principle

The `svm.core` modules are intended to remain dataset-agnostic.

They should NOT hard-code:

```text
subject IDs
task names
metadata column names
ROI choices
classifier choices
```

Instead, they provide reusable utilities used by higher-level modules such as:

```text
svm.features
svm.decode
svm.scripts
```

---

# Pipeline Role

Within the full framework, `svm.core` supports:

```text
feature extraction
    ->
metadata validation
    ->
manifest construction
    ->
label/pair construction
    ->
spatial selection
    ->
time projection
    ->
decoding
    ->
matrix/RDM construction
```

The goal is to allow raw activity, oscillatory power, and oscillatory phase representations to run through the same standardized decoding interface.

---

# Standard Feature Convention

All feature arrays must use:

```text
(freq, time, trial, channel)
```

regardless of representation type.

Examples:

```text
raw amplitude:
    (1, time, trial, channel)

power:
    (freq, time, trial, channel)

phase angle:
    (freq, time, trial, channel)

phase sin_cos:
    (2*freq, time, trial, channel)
```

This standardization enables all downstream decoding scripts to remain representation-independent.

---

# Current Supported Framework Features

Current framework capabilities include:

```text
raw decoding
power decoding
phase decoding

pairwise decoding
LOSO decoding
single-subject CV decoding

ROI-based decoding
sensor-group decoding

timepoint decoding
window decoding
sliding-window decoding

matrix generation
RDM generation
heatmap visualization
```

The framework is designed to support future extensions including:

```text
time-resolved decoding
searchlight decoding
source-space decoding
cross-condition decoding
cross-modal decoding
RSA integration
multivariate geometry analysis
```
