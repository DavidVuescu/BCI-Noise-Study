"""
Load a single BCI Noise Study recording into a coherent in-memory object.

A recording is six files on disk:
    sub-XX_cond-YY_eeg.npy         (n_samples, 17) float32 — raw EEG + aux channels
    sub-XX_cond-YY_acqtime.npy     (n_samples,) float64    — counter-derived time axis
    sub-XX_cond-YY_timestamps.npy  (n_samples,) float64    — receive-time (forensic)
    sub-XX_cond-YY_meta.json       — recorder-side metadata
    sub-XX_cond-YY_markers.csv     — one row per flash
    sub-XX_cond-YY_session.json    — stimulus-side metadata

The loader converts the EEG into an MNE Raw object and the markers into a
pandas DataFrame with each marker's sample index pre-computed. Downstream
modules (preprocess, classifier, erp) consume the Recording object.
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
        subject_id, condition: from filenames
        raw: MNE Raw with the 8 EEG channels in microvolts.
            The Unicorn streams in microvolts already (DC-coupled), but
            MNE expects volts internally, so we scale to volts before
            building Raw — MNE's filter and plotting code assumes SI units.
        markers: DataFrame with one row per flash. Has the original CSV
            columns PLUS a 'sample' column giving each marker's index into
            the EEG array (counter-aligned, jitter-free).
        meta: combined dict {recorder_meta, session_meta}. Use for
            wall_clock_anchor, dropout counts, late frames, reported counts.
        acq_time: (n_samples,) the counter-derived time axis. Mostly
            informational; MNE Raw computes its own time axis from
            sample_rate, but you can use this to sanity-check alignment.
    """
    subject_id: str
    condition: str
    raw: mne.io.Raw
    markers: pd.DataFrame
    meta: dict
    acq_time: np.ndarray

    @property
    def n_epochs_planned(self) -> int:
        """How many epochs the markers say to extract."""
        return len(self.markers)

    @property
    def duration_s(self) -> float:
        return float(self.acq_time[-1]) if len(self.acq_time) > 0 else 0.0


def _file_paths(data_dir: Path, subject_id: str, condition: str) -> dict[str, Path]:
    """Return the six file paths for one recording. Doesn't check existence."""
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
    """Build an MNE Raw object from the 8-channel EEG array.

    Args:
        eeg_array_uv: (n_samples, 8) array in microvolts as recorded.

    Returns:
        mne.io.Raw containing the 8 EEG channels in volts (MNE convention).
    """
    # MNE wants (n_channels, n_samples), we have (n_samples, n_channels).
    # MNE wants SI units (volts), we have microvolts.
    data_v = eeg_array_uv.T * 1e-6

    info = mne.create_info(
        ch_names=EEG_CHANNEL_NAMES,
        sfreq=SAMPLE_RATE_HZ,
        ch_types=["eeg"] * N_EEG_CHANNELS,
    )

    # Apply standard 10-20 montage so MNE knows where these channels live
    # in 3D space. This is what enables topographic plots, source
    # localization, and channel-neighbor operations later.
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = mne.io.RawArray(data_v, info, verbose="WARNING")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    return raw


def _attach_sample_index(markers_df: pd.DataFrame, wall_clock_anchor: float) -> pd.DataFrame:
    """Add a 'sample' column to markers, computed from wall_time and anchor.

    The wall_clock_anchor is time.time() of the FIRST received UDP sample.
    Each marker's wall_time is time.time() right after pygame's flip().
    So:
        seconds_from_recording_start = marker.wall_time - wall_clock_anchor
        sample_index = round(seconds_from_recording_start * sample_rate)

    The counter-derived acquisition grid is uniform at 4 ms per sample,
    so this rounding is sub-sample accurate. The only error is the
    constant offset between "first packet received" (wall_clock_anchor)
    and "first sample acquired" (a fixed Bluetooth-and-buffering delay,
    same for all epochs in a recording — so it shifts the time axis but
    doesn't smear it).
    """
    df = markers_df.copy()
    seconds_from_start = df["wall_time"].values - wall_clock_anchor
    df["sample"] = np.round(seconds_from_start * SAMPLE_RATE_HZ).astype(int)
    return df


def load_recording(
    subject_id: str,
    condition: str,
    data_dir: str | Path = "data/raw",
) -> Recording:
    """Load one recording into a Recording object.

    Args:
        subject_id: e.g. "pilot-self-day0"
        condition: "control" | "chewing" | "emi" | "acoustic"
        data_dir: where the six files live.

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

    # Load all six files.
    eeg = np.load(paths["eeg"])             # (n_samples, 17) float32
    acq_time = np.load(paths["acqtime"])    # (n_samples,) float64
    # timestamps loaded only if we ever want forensic timing analysis
    # ts = np.load(paths["timestamps"])

    with open(paths["meta"]) as f:
        rec_meta = json.load(f)
    with open(paths["session"]) as f:
        sess_meta = json.load(f)
    markers = pd.read_csv(paths["markers"])

    # Extract the 8 EEG channels (drop accelerometer, gyro, battery, counter, validation).
    eeg_8ch = eeg[:, :N_EEG_CHANNELS]

    # Build MNE Raw.
    raw = _build_mne_raw(eeg_8ch)

    # Align markers to sample indices.
    wall_anchor = rec_meta["wall_clock_anchor_unix"]
    markers_aligned = _attach_sample_index(markers, wall_anchor)

    # Combined metadata for convenience downstream.
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
    )
