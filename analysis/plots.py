"""
Standard visualisations for the BCI Noise Study.

All functions return a matplotlib Figure so the caller decides
whether to show(), savefig(), or embed in a notebook.

Usage:
    from analysis.plots import plot_erp, plot_confusion_matrix, ...
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mne

from analysis.preprocess import PreprocessResult


# Colour conventions used throughout
TARGET_COLOR = "#2166ac"
NONTARGET_COLOR = "#d6604d"
CONDITION_COLORS = {
    "control": "#4dac26",
    "chewing": "#d01c8b",
    "emi": "#f1b6da",
    "acoustic": "#b8e186",
    "control_heldout": "#4dac26",
}


def plot_erp(
    result: PreprocessResult,
    channel: str = "Cz",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Target vs nontarget ERP overlay for one channel.

    Marks the canonical P300 window (250-500 ms) with a shaded region.
    """
    epochs = result.epochs
    if channel not in epochs.ch_names:
        raise ValueError(f"Channel {channel!r} not in epochs. "
                         f"Available: {epochs.ch_names}")

    fig, ax = (plt.subplots(figsize=(8, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    times_ms = epochs.times * 1000

    target_epochs = epochs["target"]
    nontarget_epochs = epochs["nontarget"]

    def _mean_and_sem(ep, ch):
        data = ep.get_data(picks=[ch])[:, 0, :] * 1e6  # volts -> microvolts
        mean = data.mean(axis=0)
        sem = data.std(axis=0) / np.sqrt(len(data))
        return mean, sem

    if len(target_epochs) > 0:
        t_mean, t_sem = _mean_and_sem(target_epochs, channel)
        ax.plot(times_ms, t_mean, color=TARGET_COLOR,
                linewidth=1.8, label=f"Target (n={len(target_epochs)})")
        ax.fill_between(times_ms, t_mean - t_sem, t_mean + t_sem,
                        color=TARGET_COLOR, alpha=0.2)

    if len(nontarget_epochs) > 0:
        n_mean, n_sem = _mean_and_sem(nontarget_epochs, channel)
        ax.plot(times_ms, n_mean, color=NONTARGET_COLOR,
                linewidth=1.8, label=f"Nontarget (n={len(nontarget_epochs)})")
        ax.fill_between(times_ms, n_mean - n_sem, n_mean + n_sem,
                        color=NONTARGET_COLOR, alpha=0.2)

    # P300 window annotation
    ax.axvspan(250, 500, alpha=0.08, color="gold", label="P300 window")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)

    subject = result.rejection_log.get("subject_id", "?")
    condition = result.rejection_log.get("condition", "?")
    ax.set_title(title or f"ERP — sub-{subject}  {condition}  ch={channel}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(fontsize=9)
    ax.set_xlim(times_ms[0], times_ms[-1])
    return fig


def plot_psd(
    raw: mne.io.Raw,
    title: str | None = None,
    fmax: float = 60.0,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Power spectral density across all EEG channels.

    Useful for spotting line noise, muscle artifact, and filter roll-off.
    """
    fig, ax = (plt.subplots(figsize=(8, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    spectrum = raw.compute_psd(method="welch", fmax=fmax, verbose="WARNING")
    freqs = spectrum.freqs
    psds = spectrum.get_data()  # (n_channels, n_freqs), watts/Hz

    psds_db = 10 * np.log10(psds + 1e-30)
    mean_db = psds_db.mean(axis=0)
    std_db = psds_db.std(axis=0)

    ax.plot(freqs, mean_db, color="#333333", linewidth=1.5, label="Mean across channels")
    ax.fill_between(freqs, mean_db - std_db, mean_db + std_db,
                    color="#333333", alpha=0.2, label="±1 SD")

    ax.axvline(50, color="red", linewidth=0.8, linestyle="--", label="50 Hz notch")
    ax.axvline(1, color="blue", linewidth=0.8, linestyle="--", label="1 Hz highpass")
    ax.axvline(30, color="blue", linewidth=0.8, linestyle="--", label="30 Hz lowpass")

    ax.set_title(title or "Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, fmax)
    return fig


def plot_rejection_summary(
    rejection_log: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Stacked bar showing kept / boundary-dropped / amplitude-dropped epochs."""
    fig, ax = (plt.subplots(figsize=(6, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    n_kept = rejection_log["n_kept"]
    n_boundary = rejection_log["n_boundary_dropped"]
    n_amplitude = rejection_log["n_amplitude_dropped"]
    total = rejection_log["n_planned"]

    bars = [n_kept, n_boundary, n_amplitude]
    labels = [f"Kept ({n_kept})", f"Boundary ({n_boundary})", f"Amplitude ({n_amplitude})"]
    colors = ["#4dac26", "#f4a582", "#d6604d"]

    bottom = 0
    for val, label, color in zip(bars, labels, colors):
        ax.bar(0, val, bottom=bottom, color=color, label=label, width=0.5)
        if val > 0:
            ax.text(0, bottom + val / 2, str(val),
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        bottom += val

    subject = rejection_log.get("subject_id", "?")
    condition = rejection_log.get("condition", "?")
    rate = rejection_log.get("rejection_rate", 0) * 100
    ax.set_title(f"Epoch rejection — sub-{subject}  {condition}\n"
                 f"Total: {total}  |  Rejection rate: {rate:.1f}%")
    ax.set_xticks([])
    ax.set_ylabel("Number of epochs")
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_confusion_matrix(
    per_condition_metrics: dict,
    condition: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Confusion matrix heatmap for one condition."""
    fig, ax = (plt.subplots(figsize=(4, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    metrics = per_condition_metrics[condition]
    cm = metrics["confusion_matrix"]
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])

    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = ["Nontarget", "Target"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=14, color="black")

    bal_acc = metrics["balanced_accuracy"] * 100
    ax.set_title(f"{condition} — bal. acc. {bal_acc:.1f}%")
    return fig


def plot_condition_accuracy(
    result_dict: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Horizontal bar chart of balanced accuracy across all evaluated conditions."""
    fig, ax = (plt.subplots(figsize=(7, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    conditions = list(result_dict["per_condition"].keys())
    accuracies = [result_dict["per_condition"][c]["balanced_accuracy"] * 100
                  for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conditions]

    bars = ax.barh(conditions, accuracies, color=colors, edgecolor="white")
    ax.axvline(50, color="black", linewidth=0.8, linestyle="--", label="Chance (50%)")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10)

    subject = result_dict.get("subject_id", "?")
    clf_type = result_dict.get("classifier_type", "?")
    ax.set_title(f"Balanced accuracy by condition — sub-{subject}  [{clf_type}]")
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    return fig


def plot_subject_summary(
    result: PreprocessResult,
    result_dict: dict,
    channel: str = "Cz",
) -> plt.Figure:
    """Single-figure summary: ERP + rejection + confusion matrices (all conditions).

    Intended for a quick daily sanity-check on a freshly processed subject.
    """
    conditions = list(result_dict["per_condition"].keys())
    n_conds = len(conditions)

    fig = plt.figure(figsize=(5 * (n_conds + 2), 8), layout="constrained")
    gs = gridspec.GridSpec(2, n_conds + 2, figure=fig)

    ax_erp = fig.add_subplot(gs[0, :2])
    plot_erp(result, channel=channel, ax=ax_erp)

    ax_rej = fig.add_subplot(gs[1, 0])
    plot_rejection_summary(result.rejection_log, ax=ax_rej)

    ax_acc = fig.add_subplot(gs[1, 1])
    plot_condition_accuracy(result_dict, ax=ax_acc)

    for i, cond in enumerate(conditions):
        ax_cm = fig.add_subplot(gs[:, i + 2])
        plot_confusion_matrix(result_dict["per_condition"], cond, ax=ax_cm)

    subject = result.rejection_log.get("subject_id", "?")
    fig.suptitle(f"Subject summary — sub-{subject}", fontsize=14, fontweight="bold")
    return fig





def load_all_results(derived_root: str | Path = "data/derived") -> list[dict]:
    """Scan data/derived/classifier-v1 and load every _results.json found.

    Returns a list of result dicts, one per processed subject.
    """
    derived_root = Path(derived_root)
    clf_dir = derived_root / "classifier-v2"
    results = []
    for path in sorted(clf_dir.glob("sub-*/sub-*_results.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results
