"""Diagnostic: is the chewing artifact actually present in the raw signal,
even though preprocessing's bandpass appears to remove it?

Compares power spectral density between control and chewing on UNFILTERED
data. If chewing has visible 30-100 Hz energy vs control, the artifact is
present but filterable. If they look identical, chewing may not have
produced much EMG in the first place.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from analysis.loader import load_recording

POSTERIOR_CHANNELS = ["PO7", "Oz", "PO8"]  # channels we care about most


def main():
    subject = "pilot-self-day0"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, ch_name in zip(axes, POSTERIOR_CHANNELS):
        for cond, color in [("control", "tab:blue"), ("chewing", "tab:red")]:
            rec = load_recording(subject, cond)
            ch_idx = rec.raw.ch_names.index(ch_name)
            # raw.get_data() returns in volts; convert to uV for plotting
            data = rec.raw.get_data(picks=[ch_idx])[0] * 1e6
            sfreq = rec.raw.info["sfreq"]

            f, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 2))
            # Plot in dB µV²/Hz for log-friendly comparison
            ax.semilogy(f, psd, label=cond, color=color, alpha=0.8)

        ax.set_title(f"Channel {ch_name}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 100)
        ax.axvspan(30, 100, color="gray", alpha=0.15, label="EMG band (>30 Hz)")
        ax.axvline(30, color="black", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("PSD (µV²/Hz)")
    fig.suptitle("Power Spectral Density: Control vs Chewing (unfiltered)\n"
                 "Bandpass cuts at 30 Hz (dashed line); EMG energy concentrated >30 Hz")
    fig.tight_layout()

    out_path = "data/derived/preprocessing-v1/sub-pilot-self-day0/chewing_artifact_check.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved diagnostic to {out_path}")
    print("\nWhat to look for:")
    print("  - If chewing (red) sits ABOVE control (blue) in the 30-100 Hz")
    print("    gray band: chewing artifact is real, filter is removing it.")
    print("  - If chewing and control overlap throughout: chewing didn't")
    print("    produce much EMG (maybe pace was too gentle, or rag was too soft).")
    print("  - If chewing is BELOW control: something weird, investigate.")


if __name__ == "__main__":
    main()
