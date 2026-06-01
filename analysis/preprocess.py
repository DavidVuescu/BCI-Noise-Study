"""
Preprocessing pipeline: Recording → MNE Epochs.

Applies the filtering, epoching, baseline correction, and rejection steps
specified in pre-registration §5 and §6, in that order. Returns an MNE
Epochs object and a rejection log; optionally persists both to disk.

Output paths:
    data/derived/preprocessing-v1/sub-<id>/sub-<id>_cond-<cond>_epo.fif
    data/derived/preprocessing-v1/sub-<id>/sub-<id>_cond-<cond>_rejection.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from analysis.loader import Recording


# ---- Pre-registered parameters -----------------------------------------
# These match §5 and §6 of the pre-registration. Changing any of these
# without an OSF amendment is a deviation; log in DEVIATIONS.md.
NOTCH_FREQ_HZ = 50.0           # Romanian mains
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 30.0
FILTER_METHOD = "iir"          # IIR with iir_params -> Butterworth (registered)
EPOCH_TMIN_S = -0.200          # 200 ms pre-stimulus baseline window start
EPOCH_TMAX_S = 0.800           # 800 ms post-stimulus
BASELINE_WINDOW = (-0.200, 0.0)
REJECT_PEAK_TO_PEAK_UV = 150.0  # ±150 µV peak-to-peak rejection threshold
BOUNDARY_LEAD_S = 2.0           # reject epochs within first 2s of sub-block
BOUNDARY_TAIL_S = 1.0           # reject epochs within last 1s of sub-block

# Derived dir layout matches the BIDS-shadow convention discussed earlier.
DERIVED_PIPELINE = "preprocessing-v1"


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs.

    epochs: MNE Epochs (filtered, baseline-corrected, rejection-applied).
        Length is n_kept, not n_planned.
    rejection_log: dict summarizing what was rejected and why.
    """
    epochs: mne.Epochs
    rejection_log: dict


def _compute_subblock_boundaries(markers: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """For each sub_block_index, return (first_sample, last_sample).

    Used for boundary-rejection per pre-reg §6.
    """
    boundaries = {}
    for sb_idx, group in markers.groupby("sub_block_index"):
        boundaries[int(sb_idx)] = (int(group["sample"].min()), int(group["sample"].max()))
    return boundaries


def _mark_boundary_rejections(
    markers: pd.DataFrame,
    boundaries: dict[int, tuple[int, int]],
    sample_rate: float,
) -> np.ndarray:
    """Return a boolean array (one per marker) marking boundary-rejected epochs.

    True = reject (epoch falls within BOUNDARY_LEAD_S of sub-block start
    or BOUNDARY_TAIL_S of sub-block end).
    """
    lead_samples = int(BOUNDARY_LEAD_S * sample_rate)
    tail_samples = int(BOUNDARY_TAIL_S * sample_rate)
    reject = np.zeros(len(markers), dtype=bool)
    for i, row in markers.iterrows():
        sb_start, sb_end = boundaries[int(row["sub_block_index"])]
        sample = int(row["sample"])
        if sample < sb_start + lead_samples or sample > sb_end - tail_samples:
            reject[i] = True
    return reject


def preprocess_recording(
    rec: Recording,
    save: bool = True,
    derived_root: str | Path = "data/derived",
) -> PreprocessResult:
    """Apply filtering, epoching, baseline correction, rejection.

    Args:
        rec: Recording from loader.load_recording().
        save: If True, persist Epochs and rejection log to disk.
        derived_root: Base path for derived data.

    Returns:
        PreprocessResult with the cleaned Epochs and rejection log.
    """
    raw = rec.raw.copy()  # don't mutate the input
    sfreq = raw.info["sfreq"]

    # ---- 1. Notch filter at 50 Hz --------------------------------------
    # MNE's default notch is a notch_filter with a specified frequency.
    # Width and ripple use sensible defaults; we don't need to override.
    raw.notch_filter(
        freqs=[NOTCH_FREQ_HZ],
        method=FILTER_METHOD,
        iir_params=dict(order=4, ftype="butter"),
        verbose="WARNING",
    )

    # ---- 2. Bandpass 1-30 Hz, Butterworth, zero-phase ------------------
    # "iir" + iir_params=dict(ftype="butter") gives us a Butterworth filter,
    # matching the registered specification. method="iir" defaults to
    # zero-phase via filtfilt internally.
    raw.filter(
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        method=FILTER_METHOD,
        iir_params=dict(order=4, ftype="butter"),
        verbose="WARNING",
    )

    # ---- 3. Build events array for MNE epoching ------------------------
    # MNE wants events as (n_events, 3): [sample, 0, event_id].
    # We use event_id=1 for non-target, event_id=2 for target.
    # The 'metadata' kwarg lets us carry the full marker DataFrame
    # alongside each epoch.
    markers = rec.markers.reset_index(drop=True)
    events = np.column_stack([
        markers["sample"].astype(int).values,
        np.zeros(len(markers), dtype=int),
        np.where(markers["is_target"], 2, 1),
    ])
    event_id = {"nontarget": 1, "target": 2}

    # ---- 4. Boundary rejection -----------------------------------------
    # Computed BEFORE epoching so we can pass a reject_by_annotation
    # mechanism — easiest path is to drop these rows from events.
    boundaries = _compute_subblock_boundaries(markers)
    boundary_rejected = _mark_boundary_rejections(markers, boundaries, sfreq)
    n_boundary_dropped = int(boundary_rejected.sum())

    events_kept = events[~boundary_rejected]
    metadata_kept = markers[~boundary_rejected].reset_index(drop=True)

    # ---- 5. Epoching with baseline correction --------------------------
    # reject=dict(eeg=...) applies the peak-to-peak threshold. MNE expects
    # the threshold in volts since we converted to volts in the loader.
    reject_dict = {"eeg": REJECT_PEAK_TO_PEAK_UV * 1e-6}

    epochs = mne.Epochs(
        raw,
        events=events_kept,
        event_id=event_id,
        tmin=EPOCH_TMIN_S,
        tmax=EPOCH_TMAX_S,
        baseline=BASELINE_WINDOW,
        reject=reject_dict,
        preload=True,
        metadata=metadata_kept,
        verbose="WARNING",
    )

    # MNE silently drops rejected epochs during construction.
    # Count what survived for the rejection log.
    n_after_boundary = len(events_kept)
    n_kept = len(epochs)
    n_amplitude_dropped = n_after_boundary - n_kept

    # ---- 6. Build rejection log ----------------------------------------
    n_planned = len(markers)
    rejection_log = {
        "subject_id": rec.subject_id,
        "condition": rec.condition,
        "n_planned": n_planned,
        "n_boundary_dropped": n_boundary_dropped,
        "n_amplitude_dropped": n_amplitude_dropped,
        "n_kept": n_kept,
        "rejection_rate": (n_planned - n_kept) / n_planned if n_planned > 0 else 0.0,
        "per_subblock": {},
        "parameters": {
            "notch_hz": NOTCH_FREQ_HZ,
            "bandpass_low_hz": BANDPASS_LOW_HZ,
            "bandpass_high_hz": BANDPASS_HIGH_HZ,
            "filter": "iir_butterworth_order4",
            "epoch_tmin_s": EPOCH_TMIN_S,
            "epoch_tmax_s": EPOCH_TMAX_S,
            "baseline_window": list(BASELINE_WINDOW),
            "reject_peak_to_peak_uv": REJECT_PEAK_TO_PEAK_UV,
            "boundary_lead_s": BOUNDARY_LEAD_S,
            "boundary_tail_s": BOUNDARY_TAIL_S,
        },
    }

    # Per-sub-block breakdown
    for sb_idx in sorted(markers["sub_block_index"].unique()):
        sb_planned = int((markers["sub_block_index"] == sb_idx).sum())
        sb_kept = int((epochs.metadata["sub_block_index"] == sb_idx).sum()) \
                  if epochs.metadata is not None else 0
        rejection_log["per_subblock"][int(sb_idx)] = {
            "planned": sb_planned,
            "kept": sb_kept,
            "rejection_rate": (sb_planned - sb_kept) / sb_planned if sb_planned > 0 else 0.0,
        }

    # ---- 7. Persist if requested ---------------------------------------
    if save:
        out_dir = Path(derived_root) / DERIVED_PIPELINE / f"sub-{rec.subject_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sub-{rec.subject_id}_cond-{rec.condition}"
        epochs.save(out_dir / f"{stem}_epo.fif", overwrite=True, verbose="WARNING")
        with open(out_dir / f"{stem}_rejection.json", "w") as f:
            json.dump(rejection_log, f, indent=2)

    return PreprocessResult(epochs=epochs, rejection_log=rejection_log)
