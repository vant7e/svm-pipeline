#!/usr/bin/env python3
"""
Morlet-wavelet computation for EEG power and phase features.

This module contains Morlet-specific logic only. It does not discover or
combine sessions, build trial metadata, average observations, derive power or
phase representations, save feature packages, z-score features, or permanently
save complex coefficients.

Input Epochs must already be loaded, validated, concatenated, and sorted by
svm.features.common.

Each yielded complex chunk has shape:
    (trial, channel, frequency_chunk, time)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

from svm.features.common import (
    ANALYSIS_N_TIMEPOINTS,
    ANALYSIS_WINDOW_START_SECONDS,
    EXPECTED_SFREQ,
    OT_CHANNELS,
    find_paper_time_indices,
)

DEFAULT_FREQ_START_HZ = 2.0
DEFAULT_FREQ_STOP_HZ = 30.0
DEFAULT_FREQ_STEP_HZ = 1.0
DEFAULT_N_CYCLES_DIVISOR = 2.0
DEFAULT_FREQUENCY_CHUNK_SIZE = 5


@dataclass(frozen=True)
class WaveletConfig:
    """Immutable settings for the Morlet transformation."""

    freq_start_hz: float = DEFAULT_FREQ_START_HZ
    freq_stop_hz: float = DEFAULT_FREQ_STOP_HZ
    freq_step_hz: float = DEFAULT_FREQ_STEP_HZ
    n_cycles_divisor: float = DEFAULT_N_CYCLES_DIVISOR
    frequency_chunk_size: int = DEFAULT_FREQUENCY_CHUNK_SIZE
    analysis_window_start_seconds: float = ANALYSIS_WINDOW_START_SECONDS
    analysis_n_timepoints: int = ANALYSIS_N_TIMEPOINTS
    use_fft: bool = True
    zero_mean: bool = True
    n_jobs: int = 1

    def frequencies(self) -> np.ndarray:
        return build_frequency_vector(
            start_hz=self.freq_start_hz,
            stop_hz=self.freq_stop_hz,
            step_hz=self.freq_step_hz,
        )

    def as_dict(self) -> dict[str, Any]:
        freqs = self.frequencies()
        return {
            "freq_start_hz": float(self.freq_start_hz),
            "freq_stop_hz": float(self.freq_stop_hz),
            "freq_step_hz": float(self.freq_step_hz),
            "frequencies_hz": freqs.tolist(),
            "n_cycles_rule": "frequency / n_cycles_divisor",
            "n_cycles_divisor": float(self.n_cycles_divisor),
            "n_cycles": (freqs / self.n_cycles_divisor).tolist(),
            "frequency_chunk_size": int(self.frequency_chunk_size),
            "analysis_window_start_seconds": float(
                self.analysis_window_start_seconds
            ),
            "analysis_n_timepoints": int(self.analysis_n_timepoints),
            "use_fft": bool(self.use_fft),
            "zero_mean": bool(self.zero_mean),
            "n_jobs": int(self.n_jobs),
            "wavelet_input_cropped_before_tfr": False,
            "time_selection_applied_after_tfr": True,
            "complex_coefficients_saved_permanently": False,
        }


@dataclass(frozen=True)
class ComplexWaveletChunk:
    """One frequency chunk of complex Morlet coefficients."""

    data: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    frequency_indices: np.ndarray


def build_frequency_vector(
    start_hz: float = DEFAULT_FREQ_START_HZ,
    stop_hz: float = DEFAULT_FREQ_STOP_HZ,
    step_hz: float = DEFAULT_FREQ_STEP_HZ,
) -> np.ndarray:
    """Construct an inclusive, strictly increasing frequency vector."""

    start_hz = float(start_hz)
    stop_hz = float(stop_hz)
    step_hz = float(step_hz)

    if not np.isfinite([start_hz, stop_hz, step_hz]).all():
        raise ValueError("Frequency settings must be finite.")
    if start_hz <= 0:
        raise ValueError("start_hz must be positive.")
    if stop_hz < start_hz:
        raise ValueError("stop_hz must be >= start_hz.")
    if step_hz <= 0:
        raise ValueError("step_hz must be positive.")

    n_steps = int(np.floor(((stop_hz - start_hz) / step_hz) + 1e-12))
    freqs = start_hz + np.arange(n_steps + 1, dtype=np.float64) * step_hz

    if freqs[-1] < stop_hz - (step_hz * 1e-9):
        freqs = np.append(freqs, stop_hz)

    if not np.isclose(freqs[-1], stop_hz):
        raise ValueError(
            "The requested interval is not evenly represented by the step. "
            f"Last generated frequency: {freqs[-1]}; requested stop: {stop_hz}."
        )

    freqs[-1] = stop_hz

    if np.any(np.diff(freqs) <= 0):
        raise RuntimeError("Generated frequency vector is not increasing.")

    return freqs


def parse_frequency_string(freq_string: str) -> np.ndarray:
    """Parse start:stop:step or comma-separated frequencies."""

    freq_string = str(freq_string).strip()
    if not freq_string:
        raise ValueError("Frequency string cannot be empty.")

    if ":" in freq_string:
        parts = freq_string.split(":")
        if len(parts) != 3:
            raise ValueError("Colon syntax must be start:stop:step.")
        return build_frequency_vector(*map(float, parts))

    freqs = np.asarray(
        [float(value.strip()) for value in freq_string.split(",")],
        dtype=np.float64,
    )

    if freqs.size == 0:
        raise ValueError("No frequencies were parsed.")
    if not np.isfinite(freqs).all():
        raise ValueError("Parsed frequencies contain NaN or infinity.")
    if np.any(freqs <= 0):
        raise ValueError("All frequencies must be positive.")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("Frequencies must be strictly increasing.")

    return freqs


def validate_wavelet_epochs(
    epochs: mne.BaseEpochs,
    picks: list[str] | tuple[str, ...] = tuple(OT_CHANNELS),
) -> None:
    """Validate requirements specific to Morlet computation."""

    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            "epochs must be an MNE Epochs object, "
            f"got {type(epochs).__name__}."
        )
    if len(epochs) == 0:
        raise RuntimeError("Cannot compute wavelets for empty Epochs.")

    sfreq = float(epochs.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise RuntimeError(
            f"Expected sampling rate {EXPECTED_SFREQ} Hz, got {sfreq} Hz."
        )

    missing_channels = [ch for ch in picks if ch not in epochs.ch_names]
    if missing_channels:
        raise RuntimeError(
            f"Epochs are missing required channels: {missing_channels}"
        )

    data = epochs.get_data(picks=list(picks), copy=False)
    expected_shape = (len(epochs), len(picks), len(epochs.times))

    if data.shape != expected_shape:
        raise RuntimeError(
            f"Expected wavelet input shape {expected_shape}, got {data.shape}."
        )
    if not np.isfinite(data).all():
        raise RuntimeError("Wavelet input contains NaN or infinity.")


def validate_config(config: WaveletConfig) -> None:
    """Validate Morlet configuration."""

    freqs = config.frequencies()

    if config.n_cycles_divisor <= 0:
        raise ValueError("n_cycles_divisor must be positive.")
    if config.frequency_chunk_size <= 0:
        raise ValueError("frequency_chunk_size must be positive.")
    if config.analysis_n_timepoints <= 0:
        raise ValueError("analysis_n_timepoints must be positive.")
    if config.n_jobs == 0:
        raise ValueError("n_jobs cannot be zero.")

    n_cycles = freqs / float(config.n_cycles_divisor)
    if not np.isfinite(n_cycles).all() or np.any(n_cycles <= 0):
        raise RuntimeError(f"Invalid n_cycles values: {n_cycles.tolist()}.")


def iter_complex_wavelet_chunks(
    epochs: mne.BaseEpochs,
    config: WaveletConfig | None = None,
    picks: list[str] | tuple[str, ...] = tuple(OT_CHANNELS),
) -> Iterator[ComplexWaveletChunk]:
    """Yield complex Morlet coefficients one frequency chunk at a time."""

    if config is None:
        config = WaveletConfig()

    validate_config(config)
    picks = list(picks)
    validate_wavelet_epochs(epochs=epochs, picks=picks)

    freqs = config.frequencies()
    n_cycles = freqs / float(config.n_cycles_divisor)

    time_indices, selected_times = find_paper_time_indices(
        epochs=epochs,
        start_seconds=config.analysis_window_start_seconds,
        n_timepoints=config.analysis_n_timepoints,
    )

    epoch_data = epochs.get_data(picks=picks, copy=True)
    sfreq = float(epochs.info["sfreq"])
    n_trials, n_channels, _ = epoch_data.shape

    print(
        "[WAVELET] Input shape: "
        f"{epoch_data.shape} = (trial, channel, full_epoch_time)"
    )
    print(
        "[WAVELET] Frequencies: "
        f"{freqs[0]:g}-{freqs[-1]:g} Hz ({len(freqs)} bins)"
    )
    print(
        "[WAVELET] n_cycles rule: frequency / "
        f"{config.n_cycles_divisor:g}"
    )
    print(
        "[WAVELET] Post-TFR time selection: "
        f"{selected_times[0]:.9f}-{selected_times[-1]:.9f} s "
        f"({len(selected_times)} samples)"
    )
    print("[WAVELET] OT channel order: " + ", ".join(picks))

    for chunk_start in range(0, len(freqs), config.frequency_chunk_size):
        chunk_stop = min(
            chunk_start + config.frequency_chunk_size,
            len(freqs),
        )

        frequency_indices = np.arange(chunk_start, chunk_stop, dtype=int)
        chunk_freqs = freqs[frequency_indices]
        chunk_n_cycles = n_cycles[frequency_indices]

        print(
            "[WAVELET] Computing frequencies "
            f"{chunk_freqs[0]:g}-{chunk_freqs[-1]:g} Hz "
            f"({len(chunk_freqs)} bins)."
        )

        complex_full = tfr_array_morlet(
            data=epoch_data,
            sfreq=sfreq,
            freqs=chunk_freqs,
            n_cycles=chunk_n_cycles,
            zero_mean=config.zero_mean,
            use_fft=config.use_fft,
            decim=1,
            output="complex",
            n_jobs=config.n_jobs,
            verbose=False,
        )

        expected_full_shape = (
            n_trials,
            n_channels,
            len(chunk_freqs),
            len(epochs.times),
        )
        if complex_full.shape != expected_full_shape:
            raise RuntimeError(
                f"Expected complete TFR shape {expected_full_shape}, "
                f"got {complex_full.shape}."
            )

        complex_selected = complex_full[:, :, :, time_indices].astype(
            np.complex64,
            copy=False,
        )

        expected_selected_shape = (
            n_trials,
            n_channels,
            len(chunk_freqs),
            config.analysis_n_timepoints,
        )
        if complex_selected.shape != expected_selected_shape:
            raise RuntimeError(
                f"Expected selected TFR shape {expected_selected_shape}, "
                f"got {complex_selected.shape}."
            )
        if not np.isfinite(complex_selected).all():
            raise RuntimeError(
                "Complex Morlet output contains NaN or infinity for "
                f"frequencies {chunk_freqs.tolist()}."
            )

        yield ComplexWaveletChunk(
            data=complex_selected,
            freqs=chunk_freqs.copy(),
            times=selected_times.copy(),
            frequency_indices=frequency_indices,
        )

        del complex_selected
        del complex_full

    del epoch_data


def compute_complex_wavelet(
    epochs: mne.BaseEpochs,
    config: WaveletConfig | None = None,
    picks: list[str] | tuple[str, ...] = tuple(OT_CHANNELS),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all complex chunks in memory; use only for debugging."""

    if config is None:
        config = WaveletConfig()

    chunks: list[np.ndarray] = []
    returned_freqs: list[np.ndarray] = []
    selected_times: np.ndarray | None = None

    for chunk in iter_complex_wavelet_chunks(
        epochs=epochs,
        config=config,
        picks=picks,
    ):
        chunks.append(chunk.data)
        returned_freqs.append(chunk.freqs)

        if selected_times is None:
            selected_times = chunk.times
        elif not np.array_equal(selected_times, chunk.times):
            raise RuntimeError("Time vectors differ across frequency chunks.")

    if not chunks or selected_times is None:
        raise RuntimeError("No complex wavelet chunks were produced.")

    complex_feature = np.concatenate(chunks, axis=2).astype(
        np.complex64,
        copy=False,
    )
    freqs = np.concatenate(returned_freqs).astype(np.float64, copy=False)

    return complex_feature, freqs, selected_times


def build_wavelet_audit_metadata(
    config: WaveletConfig,
    selected_times: np.ndarray,
    picks: list[str] | tuple[str, ...] = tuple(OT_CHANNELS),
) -> dict[str, Any]:
    """Build Morlet-specific audit metadata."""

    selected_times = np.asarray(selected_times, dtype=np.float64)

    if selected_times.shape != (config.analysis_n_timepoints,):
        raise ValueError(
            f"Expected selected_times shape "
            f"({config.analysis_n_timepoints},), got {selected_times.shape}."
        )

    metadata = config.as_dict()
    metadata.update(
        {
            "channel_order": list(picks),
            "n_channels": len(picks),
            "selected_time_start_seconds": float(selected_times[0]),
            "selected_time_end_seconds": float(selected_times[-1]),
            "n_timepoints": int(len(selected_times)),
            "standard_saved_feature_schema": (
                "frequency, time, trial, channel"
            ),
        }
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test complex Morlet computation on one Epochs FIF file. "
            "No complex coefficients are saved."
        )
    )
    parser.add_argument("--epochs-file", type=Path, required=True)
    parser.add_argument("--freqs", default="2:30:1")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_FREQUENCY_CHUNK_SIZE,
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs_file = args.epochs_file.resolve()

    if not epochs_file.is_file():
        raise FileNotFoundError(
            f"Epochs file does not exist: {epochs_file}"
        )

    freqs = parse_frequency_string(args.freqs)

    if len(freqs) == 1:
        freq_step = 1.0
    else:
        differences = np.diff(freqs)
        if not np.allclose(differences, differences[0]):
            raise ValueError(
                "The smoke-test CLI requires equally spaced frequencies."
            )
        freq_step = float(differences[0])

    config = WaveletConfig(
        freq_start_hz=float(freqs[0]),
        freq_stop_hz=float(freqs[-1]),
        freq_step_hz=freq_step,
        frequency_chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
    )

    print(f"[LOAD] {epochs_file}")
    epochs = mne.read_epochs(
        epochs_file,
        preload=True,
        verbose="INFO",
    )

    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive.")
        epochs = epochs[: min(args.max_epochs, len(epochs))]

    total_chunks = 0
    total_frequency_bins = 0
    selected_times: np.ndarray | None = None

    for chunk in iter_complex_wavelet_chunks(
        epochs=epochs,
        config=config,
        picks=OT_CHANNELS,
    ):
        total_chunks += 1
        total_frequency_bins += len(chunk.freqs)
        selected_times = chunk.times

        print(
            "[CHUNK OK] "
            f"freqs={chunk.freqs.tolist()}, "
            f"shape={chunk.data.shape}, "
            f"dtype={chunk.data.dtype}, "
            f"size={chunk.data.nbytes / (1024 ** 2):.2f} MiB"
        )

    if selected_times is None:
        raise RuntimeError("Smoke test produced no chunks.")

    print("\n" + "=" * 80)
    print("WAVELET SMOKE TEST COMPLETED")
    print(f"Epochs: {len(epochs)}")
    print(f"Channels: {len(OT_CHANNELS)}")
    print(f"Frequency bins: {total_frequency_bins}")
    print(f"Frequency chunks: {total_chunks}")
    print(
        "Selected times: "
        f"{selected_times[0]:.9f}-{selected_times[-1]:.9f} s "
        f"({len(selected_times)} samples)"
    )
    print("Complex coefficients were not saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
