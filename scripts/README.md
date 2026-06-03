# `svm.scripts`

`svm.scripts` contains CLI entry points for the general neural decoding framework.

These scripts are intended to provide task-level execution wrappers around the reusable modules inside:

```text
svm.core
svm.features
svm.decode
```

The scripts are designed to remain dataset-general and metadata-driven.

---

# Scripts

## `extract_features.py`

Extracts standardized neural feature packages from epoched neural recordings.

This script converts MNE Epochs into standardized feature arrays with shape:

```text
(freq, time, trial, channel)
```

The resulting feature packages can later be used for:

```text
pairwise decoding
time-resolved decoding
RSA
classification
representation analysis
```

---

# Supported Feature Spaces

## Raw Features

```text
feature_space = raw
```

Supported representations:

```text
amplitude
```

Description:

```text
Raw time-domain neural activity.
A fake frequency axis of length 1 is added for compatibility with power and phase features.
```

Output shape:

```text
(1, time, trial, channel)
```

---

## Power Features

```text
feature_space = power
```

Supported representations:

```text
magnitude
magnitude_squared
log_power
```

Computed using:

```text
Morlet wavelet time-frequency decomposition
```

Output shape:

```text
(freq, time, trial, channel)
```

---

## Phase Features

```text
feature_space = phase
```

Supported representations:

```text
angle
sin_cos
complex_unit
```

Description:

```text
angle:
    raw circular phase angle in radians

sin_cos:
    circular-safe real-valued phase representation

complex_unit:
    complex unit representation exp(i * phase)
```

Output shape:

```text
(freq, time, trial, channel)
```

---

# Metadata Handling

All metadata columns from the input CSV are automatically stored as labels.

This keeps the framework task-general.

Example metadata columns:

```text
image
category
OLD_NEW
Confidence
RT
run
block
stimulus_identity
semantic_label
```

These labels can later be used for:

```text
pairwise decoding
grouping
cross-validation
RSA
trial filtering
```

---

# Time Cropping

Time cropping is controlled through:

```bash
--tmin
--tmax
```

For raw features:

```text
epochs are cropped before extraction
```

For power and phase features:

```text
Morlet TFR is computed first,
then the feature tensor is cropped afterward.

This avoids short-signal wavelet errors at low frequencies.
```

---

# Frequency Configuration

Power and phase features support flexible frequency specification.

Examples:

```text
2:30:1
```

means:

```text
2 Hz to 30 Hz in steps of 1 Hz
```

Or explicit frequencies:

```text
4,8,12,30
```

---

# Output Files

Each extraction creates:

```text
feature.npy
feature.meta.json
```

The metadata JSON stores:

```text
feature_space
representation
times
freqs
channel names
trial labels
source epoch file
time range
feature notes
```

---

# Important Note About File Naming

The framework does NOT automatically encode time ranges or frequency settings into filenames.

Therefore, when extracting multiple versions of the same feature space, users are strongly encouraged to use:

```bash
--suffix
```

Example:

```bash
--suffix t0.0-0.6
```

or:

```bash
--suffix freq2-30-1_t0.0-0.6
```

This prevents accidental overwriting of feature packages generated with different settings.

---

# Example Usage

## Raw feature extraction

```bash
python -m svm.scripts.extract_features \
  --epoch_path /path/to/10665-epo.fif \
  --metadata_path /path/to/10665-metadata.csv \
  --out_dir /path/to/features/raw \
  --subject 10665 \
  --task recognition \
  --feature_space raw \
  --tmin 0.0 \
  --tmax 0.6 \
  --suffix t0.0-0.6
```

---

## Power feature extraction

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
  --tmax 0.6 \
  --suffix freq2-30-1_t0.0-0.6
```

---

## Phase feature extraction

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
  --tmax 0.6 \
  --suffix freq2-30-1_t0.0-0.6
```

---

# Framework Role

Within the full decoding pipeline:

```text
MNE Epochs
    ->
extract_features.py
    ->
standardized feature package
    ->
projection
    ->
classifier matrix
    ->
decoding / RSA / analysis
```

The extraction stage is intentionally separated from decoding so that the same feature packages can later be reused across multiple downstream analyses.
