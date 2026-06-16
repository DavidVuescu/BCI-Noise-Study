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
    """Scan data/derived/classifier-v3 and load every _results.json found.

    Returns a list of result dicts, one per processed subject.
    """
    derived_root = Path(derived_root)
    clf_dir = derived_root / "classifier-v3"
    results = []
    for path in sorted(clf_dir.glob("sub-*/sub-*_results.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# N170 plots
# ---------------------------------------------------------------------------

# Window and channel constants mirrored from n170.py to avoid circular imports
_N170_CHANNELS = ["PO7", "Oz", "PO8"]
_N170_TMIN_MS = 130.0
_N170_TMAX_MS = 200.0


def plot_n170_erp_overlay(
    n170_result: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-condition posterior ERP overlay for one subject.

    Plots the grand-average waveform at the mean of PO7/Oz/PO8 for each
    condition, highlighting the pre-registered N170 window (130–200 ms).

    Args:
        n170_result: dict returned by run_n170_subject() or loaded from JSON.
    """
    fig, ax = (plt.subplots(figsize=(9, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    for cond, cdata in n170_result["per_condition"].items():
        times_ms = np.array(cdata["evoked_times_s"]) * 1000
        channels = cdata["evoked_per_channel_uv"]
        # Mean across the three posterior channels
        waveform = np.mean([channels[ch] for ch in _N170_CHANNELS if ch in channels], axis=0)
        color = CONDITION_COLORS.get(cond, "#888888")
        ax.plot(times_ms, waveform, color=color, linewidth=1.8, label=cond)

    ax.axvspan(_N170_TMIN_MS, _N170_TMAX_MS, alpha=0.12, color="gold",
               label=f"N170 window ({_N170_TMIN_MS:.0f}–{_N170_TMAX_MS:.0f} ms)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)

    subject = n170_result.get("subject_id", "?")
    ax.set_title(f"N170 — sub-{subject}  posterior mean (PO7/Oz/PO8)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(fontsize=9)
    return fig


def plot_n170_amplitude_bar(
    n170_result: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart of N170 amplitude per condition for one subject.

    More negative = larger N170 (expected). Noise making amplitude less
    negative is the predicted direction.
    """
    fig, ax = (plt.subplots(figsize=(6, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    from analysis.n170 import CONDITIONS  # local import to avoid circular dep
    conds = [c for c in CONDITIONS if c in n170_result["per_condition"]]
    amps = [n170_result["per_condition"][c]["n170_amplitude_uv"] for c in conds]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conds]

    bars = ax.bar(conds, amps, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)

    for bar, amp in zip(bars, amps):
        va = "bottom" if amp >= 0 else "top"
        offset = 0.05 if amp >= 0 else -0.05
        ax.text(bar.get_x() + bar.get_width() / 2,
                amp + offset, f"{amp:.2f}", ha="center", va=va, fontsize=9)

    subject = n170_result.get("subject_id", "?")
    ax.set_title(f"N170 amplitude — sub-{subject}")
    ax.set_ylabel("Mean amplitude in 130–200 ms (µV)")
    return fig


def plot_n170_group_amplitude(
    all_results: list[dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Group N170 amplitude: mean ± SD bars with individual subject dots.

    Args:
        all_results: list of dicts from load_all_n170_results().
    """
    fig, ax = (plt.subplots(figsize=(7, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    from analysis.n170 import CONDITIONS

    cond_data: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for r in all_results:
        for cond in CONDITIONS:
            if cond in r["per_condition"]:
                cond_data[cond].append(r["per_condition"][cond]["n170_amplitude_uv"])

    conds = [c for c in CONDITIONS if cond_data[c]]
    means = [float(np.mean(cond_data[c])) for c in conds]
    sds = [float(np.std(cond_data[c], ddof=1)) for c in conds]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conds]

    ax.bar(conds, means, yerr=sds, capsize=5, color=colors, edgecolor="white")

    for i, cond in enumerate(conds):
        vals = cond_data[cond]
        ax.scatter([i] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.5)
    n = len(all_results)
    ax.set_title(f"N170 amplitude by condition — n={n} subjects (mean ± SD)")
    ax.set_ylabel("Mean amplitude in 130–200 ms (µV)")
    return fig


def plot_n170_group_erp(
    all_results: list[dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Group grand-average N170 ERP: mean waveform per condition across subjects.

    Args:
        all_results: list of dicts from load_all_n170_results().
    """
    fig, ax = (plt.subplots(figsize=(9, 4), layout="constrained") if ax is None
               else (ax.get_figure(), ax))

    from analysis.n170 import CONDITIONS

    for cond in CONDITIONS:
        subject_waveforms = []
        times_ms = None
        for r in all_results:
            if cond not in r["per_condition"]:
                continue
            cdata = r["per_condition"][cond]
            if times_ms is None:
                times_ms = np.array(cdata["evoked_times_s"]) * 1000
            channels = cdata["evoked_per_channel_uv"]
            subj_wave = np.mean(
                [channels[ch] for ch in _N170_CHANNELS if ch in channels], axis=0
            )
            subject_waveforms.append(subj_wave)

        if not subject_waveforms or times_ms is None:
            continue
        waveforms_arr = np.array(subject_waveforms)
        grand_mean = waveforms_arr.mean(axis=0)
        grand_sem = waveforms_arr.std(axis=0) / np.sqrt(len(waveforms_arr))

        color = CONDITION_COLORS.get(cond, "#888888")
        ax.plot(times_ms, grand_mean, color=color, linewidth=1.8,
                label=f"{cond} (n={len(subject_waveforms)})")
        ax.fill_between(times_ms, grand_mean - grand_sem, grand_mean + grand_sem,
                        color=color, alpha=0.15)

    ax.axvspan(_N170_TMIN_MS, _N170_TMAX_MS, alpha=0.12, color="gold",
               label=f"N170 window")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)

    n = len(all_results)
    ax.set_title(f"Group grand-average N170 ERP — n={n} subjects (mean ± SEM)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(fontsize=9)
    return fig
