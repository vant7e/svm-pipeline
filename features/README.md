# `svm.features`

`svm.features` contains feature extraction modules for the general neural decoding framework.

These modules convert preprocessed MNE `Epochs` objects into standardized feature arrays that can be used by the downstream decoding pipeline.

All feature arrays follow the same shape convention:

```text
(freq, time, trial, channel)
```

This shared format allows raw activity, oscillatory power, and phase representations to be passed into the same projection and decoding interface.

---

# Modules

## `raw.py`

Extracts raw time-domain amplitude features from preprocessed epoched data.

Input:

```text
MNE Epochs object
```

MNE epoch data shape:

```text
(trial, channel, time)
```

Output:

```text
(1, time, trial, channel)
```

The frequency dimension is a fake axis of length 1:

```text
freq = [0.0]
```

This keeps raw activity compatible with power and phase features.

Main utilities:

```python
compute_raw_feature()

compute_and_save_raw_feature()

load_raw_feature()
```

Supported representation:

```text
amplitude
```

Notes:

```text
Raw features here refer to preprocessed epoched sensor-level activity.
They are not unprocessed continuous raw recordings.
```

---

## `power.py`

Computes oscillatory power features using Morlet time-frequency decomposition.

Input:

```text
MNE Epochs object
```

Output:

```text
(freq, time, trial, channel)
```

Supported representations:

```text
magnitude
magnitude_squared
log_power
```

Main utilities:

```python
parse_freqs()

compute_power_feature()

compute_and_save_power_feature()

load_power_feature()
```

Frequency strings can be specified as:

```text
2:30:1
```

which means:

```text
2 Hz to 30 Hz, step size 1 Hz
```

or as an explicit list:

```text
4,8,12
```

Morlet settings:

```text
n_cycles = freqs / 2
```

Important implementation detail:

```text
Power features are computed on the full epoch first.
Time cropping is applied after TFR computation.
```

This avoids errors caused by low-frequency wavelets being longer than a short cropped signal.

Example error avoided:

```text
wavelet longer than signal
```

---

## `phase.py`

Computes oscillatory phase features using Morlet time-frequency decomposition.

Input:

```text
MNE Epochs object
```

Output:

```text
(freq, time, trial, channel)
```

or, for `sin_cos`:

```text
(2 * freq, time, trial, channel)
```

Supported representations:

```text
angle
sin_cos
complex_unit
```

Main utilities:

```python
parse_freqs()

crop_feature_time()

compute_phase_feature()

compute_and_save_phase_feature()

load_phase_feature()
```

Representations:

```text
angle:
    phase angle in radians, range [-pi, pi]

sin_cos:
    circular-safe real-valued representation
    first half of frequency axis = sin(phase)
    second half of frequency axis = cos(phase)

complex_unit:
    complex unit representation exp(1j * phase)
```

Important implementation detail:

```text
Phase features are computed on the full epoch first.
Time cropping is applied after TFR computation.
```

This avoids short-signal wavelet length errors for low-frequency phase extraction.

---

# Standard Feature Package

Each feature module saves a standardized feature package:

```text
feature.npy
feature.meta.json
```

The `.npy` file stores the feature array.

The `.meta.json` file stores:

```text
subject
task
feature_space
representation
shape
freqs
times
ch_names
trial_ids
labels
source_epoch_file
notes
```

This metadata allows downstream scripts to load features without hard-coded assumptions.

---

# Supported Feature Spaces

Current feature spaces:

```text
raw
power
phase
```

Current representations:

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

---

# Shared Shape Convention

All features use:

```text
(freq, time, trial, channel)
```

Examples:

```text
raw amplitude:
    (1, time, trial, channel)

power magnitude_squared:
    (freq, time, trial, channel)

phase angle:
    (freq, time, trial, channel)

phase sin_cos:
    (2 * freq, time, trial, channel)

phase complex_unit:
    (freq, time, trial, channel)
```

---

# Downstream Compatibility

The output of `svm.features` is designed to be used by:

```text
svm.core.manifest
svm.decode.projection
svm.decode.pairwise
svm.scripts.run_pairwise_decoding
svm.scripts.run_time_resolved_decoding
```

Feature extraction does not perform:

```text
classifier training
RSA calculation
frequency-band averaging
pairwise decoding
statistical testing
matrix/RDM construction
```

Those steps should happen downstream.

---

# Design Principle

`svm.features` should remain representation-focused and task-general.

It should not hard-code:

```text
subject IDs
task-specific labels
image names
category names
ROI definitions
classifier choices
decoding methods
```

Instead, it should only convert neural data into standardized feature tensors and metadata.

---

# Example Workflow

Raw feature extraction:

```bash
python -m svm.scripts.extract_features \
  --epoch_path /path/to/10665-epo.fif \
  --metadata_path /path/to/10665-metadata.csv \
  --out_dir /path/to/features/raw \
  --subject 10665 \
  --task recognition \
  --feature_space raw \
  --tmin 0.0 \
  --tmax 0.6
```

Power feature extraction:

```bash
python -m svm.scripts.extract_features \
  --epoch_path /path/to/10665-epo.fif \
  --metadata_path /path/to/10665-metadata.csv \
  --out_dir /path/to/features/power \
  --subject 10665 \
  --task recognition \
  --feature_space power \
  --representation magnitude_squared \
  --freq_str 2:30:1 \
  --tmin 0.0 \
  --tmax 0.6
```

Phase feature extraction:

```bash
python -m svm.scripts.extract_features \
  --epoch_path /path/to/10665-epo.fif \
  --metadata_path /path/to/10665-metadata.csv \
  --out_dir /path/to/features/phase \
  --subject 10665 \
  --task recognition \
  --feature_space phase \
  --representation sin_cos \
  --freq_str 2:30:1 \
  --tmin 0.0 \
  --tmax 0.6
```

---

# Current Limitations

Full time-frequency features can be very large.

For example:

```text
29 frequencies × 721 time points × 325 trials × 274 channels
```

can require several GB per subject.

For large-scale power or phase analyses, consider:

```text
smaller frequency ranges
shorter time ranges
frequency-band averaging
time-window projection
chunked extraction
HPC execution
memory mapping
```

---

# Framework Role

Within the full decoding framework:

```text
MNE Epochs
    ->
svm.features
    ->
standardized feature package
    ->
svm.core.manifest
    ->
svm.decode.projection
    ->
svm.decode.pairwise
    ->
results / matrices / RDMs
```

The purpose of `svm.features` is to make different neural representations comparable through one common data structure.
