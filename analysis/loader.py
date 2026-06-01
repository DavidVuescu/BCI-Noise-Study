"""
Load a single BCI Noise Study recording into a coherent in-memory object.

A recording is six files on disk:
    sub-XX_cond-YY_eeg.npy         (n_samples, 17) float32 — raw EEG + aux channels
    sub-XX_cond-YY_acqtime.npy     (n_samples,) float64    — counter-derived time axis
    sub-XX_cond-YY_timestamps.npy  (n_samples,) float64    — receive-time wall clock
    sub-XX_cond-YY_meta.json       — recorder-side metadata
    sub-XX_cond-YY_markers.csv     — one row per flash
    sub-XX_cond-YY_session.json    — stimulus-side metadata

The loader converts the EEG into an MNE Raw object and the markers into a
pandas DataFrame with each marker's sample index pre-computed.

# ALIGNMENT APPROACH
---------------------------------------------------------------------------
The Unicorn streams via Bluetooth UDP. Two timing systems are in play:

  - Device counter (acq_time): uniform 4 ms grid, jitter-free, ground truth
    for the EEG sample axis. Derived from the hardware counter channel.

  - Wall clock (timestamps): time.time() recorded when each UDP packet is
    received. Subject to OS scheduling jitter AND a systematic linear drift
    relative to acq_time, caused by a ~1.2–1.4 ms/s mismatch between the
    host system clock and the Unicorn's internal crystal oscillator.

Measured drift across all pilot sessions: ~1.2–1.4 ms/s, accumulating to
400–870 ms over a 5–10 minute recording. Using a fixed latency correction
(as in earlier loader versions) fails because the drift magnitude varies
per session depending on how long the device had been running.

THE FIX: linear regression of acq_time on wall-clock timestamps.
For each recording we fit:

    acq_time[i]  ≈  m * (timestamps[i] - wall_anchor) + c

This absorbs both the constant Bluetooth pipeline delay and the per-session
clock drift in a single data-driven correction. We then apply the inverse
map to convert each marker's wall_time into an acq_time, and from there
to a sample index.

This approach requires no hardcoded constants and is self-calibrating from
the UDP receive log that the recorder already saves to disk.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# Channel layout from the Unicorn UDP protocol (matches recorder.py).
CHANNEL_NAMES = [
    "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8",
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "Battery", "Counter", "Validation",
]
EEG_CHANNEL_NAMES = CHANNEL_NAMES[:8]
N_EEG_CHANNELS = 8
SAMPLE_RATE_HZ = 250


@dataclass
class Recording:
    """One recording, fully loaded and aligned.

    Attributes:
        subject_id, condition: from filenames.
        raw: MNE Raw with the 8 EEG channels in volts (MNE convention).
            The Unicorn streams in microvolts; we scale on load.
        markers: DataFrame with one row per flash. Has the original CSV
            columns PLUS:
              'acq_time'  — float64, clock-corrected device time (seconds)
              'sample'    — int, index into the EEG sample array
        meta: combined dict {recorder_meta, session_meta}.
        acq_time: (n_samples,) counter-derived device time axis in seconds.
        clock_fit: (m, c) coefficients of the linear fit
                   acq_time = m * ts_relative + c
                   Stored for downstream inspection / deviation logging.
    """
    subject_id: str
    condition: str
    raw: mne.io.Raw
    markers: pd.DataFrame
    meta: dict
    acq_time: np.ndarray
    clock_fit: tuple[float, float]

    @property
    def n_epochs_planned(self) -> int:
        return len(self.markers)

    @property
    def duration_s(self) -> float:
        return float(self.acq_time[-1]) if len(self.acq_time) > 0 else 0.0


def _file_paths(data_dir: Path, subject_id: str, condition: str) -> dict[str, Path]:
    subj_dir = data_dir / f"sub-{subject_id}"
    stem = f"sub-{subject_id}_cond-{condition}"
    return {
        "eeg":        subj_dir / f"{stem}_eeg.npy",
        "acqtime":    subj_dir / f"{stem}_acqtime.npy",
        "timestamps": subj_dir / f"{stem}_timestamps.npy",
        "meta":       subj_dir / f"{stem}_meta.json",
        "markers":    subj_dir / f"{stem}_markers.csv",
        "session":    subj_dir / f"{stem}_session.json",
    }


def _build_mne_raw(eeg_array_uv: np.ndarray) -> mne.io.Raw:
    """Build an MNE Raw object from the 8-channel EEG array (in microvolts)."""
    data_v = eeg_array_uv.T * 1e-6  # (n_channels, n_samples), volts
    info = mne.create_info(
        ch_names=EEG_CHANNEL_NAMES,
        sfreq=SAMPLE_RATE_HZ,
        ch_types=["eeg"] * N_EEG_CHANNELS,
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = mne.io.RawArray(data_v, info, verbose="WARNING")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    return raw


def _fit_clock(
    timestamps: np.ndarray,
    acq_time: np.ndarray,
    wall_anchor: float,
) -> tuple[float, float]:
    """Fit a linear map from wall-clock timestamps to device acq_time.

    The Unicorn's hardware counter drifts ~1.2–1.4 ms/s relative to the
    host system clock. A linear fit over all received packets absorbs both
    the constant Bluetooth pipeline delay and this per-session drift.

    Args:
        timestamps:  (n_samples,) wall-clock receive times (time.time()).
        acq_time:    (n_samples,) counter-derived device time (seconds from 0).
        wall_anchor: time.time() of the first received packet (= timestamps[0]).

    Returns:
        (m, c) such that acq_time ≈ m * (timestamps - wall_anchor) + c
    """
    ts_relative = timestamps - wall_anchor
    m, c = np.polyfit(ts_relative, acq_time, 1)
    return float(m), float(c)


def _attach_sample_index(
    markers_df: pd.DataFrame,
    wall_anchor: float,
    clock_fit: tuple[float, float],
    n_samples: int,
) -> pd.DataFrame:
    """Add 'acq_time' and 'sample' columns to markers using the clock fit.

    Converts each marker's wall_time to device acq_time via the linear map,
    then rounds to the nearest sample index.

    Args:
        markers_df:  Raw markers DataFrame with a 'wall_time' column.
        wall_anchor: time.time() of the first received UDP packet.
        clock_fit:   (m, c) from _fit_clock.
        n_samples:   Length of the EEG array (for clipping out-of-bounds).

    Returns:
        Copy of markers_df with two new columns added.
    """
    m, c = clock_fit
    df = markers_df.copy()
    ts_relative = df["wall_time"].values - wall_anchor
    marker_acq = m * ts_relative + c
    df["acq_time"] = marker_acq
    df["sample"] = np.clip(
        np.round(marker_acq * SAMPLE_RATE_HZ).astype(int),
        0, n_samples - 1,
    )
    return df


def load_recording(
    subject_id: str,
    condition: str,
    data_dir: str | Path = "data/raw",
    verbose_clock: bool = False,
) -> Recording:
    """Load one recording into a Recording object.

    Args:
        subject_id:    e.g. "pilot-self-day0"
        condition:     "control" | "chewing" | "emi" | "acoustic"
        data_dir:      root directory containing sub-<id>/ folders.
        verbose_clock: if True, print clock-fit diagnostics to stdout.

    Returns:
        Recording object ready for downstream analysis.

    Raises:
        FileNotFoundError: if any of the six files is missing.
    """
    data_dir = Path(data_dir)
    paths = _file_paths(data_dir, subject_id, condition)

    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Recording sub-{subject_id}_cond-{condition} is missing files: "
            f"{missing}. Looked in {data_dir.absolute()}"
        )

    eeg = np.load(paths["eeg"])              # (n_samples, 17) float32
    acq_time = np.load(paths["acqtime"])     # (n_samples,) float64
    timestamps = np.load(paths["timestamps"])  # (n_samples,) float64

    with open(paths["meta"]) as f:
        rec_meta = json.load(f)
    with open(paths["session"]) as f:
        sess_meta = json.load(f)
    markers = pd.read_csv(paths["markers"])

    wall_anchor = rec_meta["wall_clock_anchor_unix"]
    n_samples = len(acq_time)

    # --- Clock calibration -------------------------------------------
    clock_fit = _fit_clock(timestamps, acq_time, wall_anchor)
    m, c = clock_fit

    if verbose_clock:
        ts_rel = timestamps - wall_anchor
        predicted = m * ts_rel + c
        residuals = acq_time - predicted
        drift_rate_ms_per_s = (1.0 - m) * 1000.0
        print(
            f"[loader] clock fit for {subject_id}/{condition}: "
            f"m={m:.6f}, c={c*1000:.2f}ms | "
            f"drift={drift_rate_ms_per_s:.3f}ms/s | "
            f"residual std={residuals.std()*1000:.2f}ms"
        )

    # --- Alignment ---------------------------------------------------
    eeg_8ch = eeg[:, :N_EEG_CHANNELS]
    raw = _build_mne_raw(eeg_8ch)
    markers_aligned = _attach_sample_index(markers, wall_anchor, clock_fit, n_samples)

    combined_meta = {
        "recorder": rec_meta,
        "session": sess_meta,
    }

    return Recording(
        subject_id=subject_id,
        condition=condition,
        raw=raw,
        markers=markers_aligned,
        meta=combined_meta,
        acq_time=acq_time,
        clock_fit=clock_fit,
    )
