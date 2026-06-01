"""Diagnostic: do target epochs look different from non-target epochs?

If the averaged target waveform differs visibly from the averaged
non-target waveform on parietal/central channels, the P300 exists in
the data and the classifier should be able to find it. If they look
identical, either there's no P300 in this recording, or labels are
scrambled.
"""
import matplotlib.pyplot as plt
import mne
import numpy as np
from pathlib import Path


SUBJECT = "pilot-self-day0"
DERIVED = Path("data/derived/preprocessing-v1") / f"sub-{SUBJECT}"


def plot_condition(cond: str, ax_p300, ax_n170):
    """Plot target vs nontarget averaged ERP for one condition.

    P300 expected: positive deflection ~250-450ms post-flash, central/parietal channels
                   (we look at Cz and Pz, channels 2 and 4).
    N170 expected: negative deflection ~130-200ms, posterior channels (PO7/Oz/PO8).
    """
    path = DERIVED / f"sub-{SUBJECT}_cond-{cond}_epo.fif"
    epochs = mne.read_epochs(path, preload=True, verbose="WARNING")

    target_meta = epochs.metadata["is_target"].values.astype(bool)
    targets = epochs[target_meta]
    nontargets = epochs[~target_meta]

    # Average over epochs to get the evoked response per channel
    tgt_evoked = targets.average()
    nt_evoked = nontargets.average()

    # Times in ms
    times_ms = tgt_evoked.times * 1000

    # P300 panel: mean of Cz and Pz (channels 2 and 4 = "Cz", "Pz")
    cz_idx = tgt_evoked.ch_names.index("Cz")
    pz_idx = tgt_evoked.ch_names.index("Pz")
    tgt_p300 = (tgt_evoked.data[cz_idx] + tgt_evoked.data[pz_idx]) / 2 * 1e6  # to uV
    nt_p300 = (nt_evoked.data[cz_idx] + nt_evoked.data[pz_idx]) / 2 * 1e6

    ax_p300.plot(times_ms, tgt_p300, label=f"{cond} target (n={len(targets)})", color="tab:red")
    ax_p300.plot(times_ms, nt_p300, label=f"{cond} non-target (n={len(nontargets)})",
                 color="tab:blue", alpha=0.7)
    ax_p300.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax_p300.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax_p300.axvspan(250, 450, color="yellow", alpha=0.15, label="P300 window")
    ax_p300.set_xlabel("Time (ms)")
    ax_p300.set_ylabel("µV (avg Cz, Pz)")
    ax_p300.set_title(f"{cond}: P300 region (Cz+Pz)")
    ax_p300.legend(fontsize=8)
    ax_p300.grid(alpha=0.3)

    # N170 panel: mean of PO7, Oz, PO8
    po7 = tgt_evoked.ch_names.index("PO7")
    oz = tgt_evoked.ch_names.index("Oz")
    po8 = tgt_evoked.ch_names.index("PO8")
    tgt_n170 = np.mean(tgt_evoked.data[[po7, oz, po8]], axis=0) * 1e6
    nt_n170 = np.mean(nt_evoked.data[[po7, oz, po8]], axis=0) * 1e6

    ax_n170.plot(times_ms, tgt_n170, label=f"target", color="tab:red")
    ax_n170.plot(times_ms, nt_n170, label=f"non-target", color="tab:blue", alpha=0.7)
    ax_n170.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax_n170.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax_n170.axvspan(130, 200, color="yellow", alpha=0.15, label="N170 window")
    ax_n170.set_xlabel("Time (ms)")
    ax_n170.set_ylabel("µV (avg PO7,Oz,PO8)")
    ax_n170.set_title(f"{cond}: N170 region (posterior)")
    ax_n170.legend(fontsize=8)
    ax_n170.grid(alpha=0.3)


def main():
    conditions = ["control", "chewing", "emi", "acoustic"]
    fig, axes = plt.subplots(len(conditions), 2, figsize=(14, 12))

    for i, cond in enumerate(conditions):
        plot_condition(cond, axes[i, 0], axes[i, 1])

    fig.suptitle(f"Target vs Non-target ERPs — {SUBJECT}\n"
                 f"If red and blue overlap, no signal to classify",
                 fontsize=13)
    fig.tight_layout()

    out_path = DERIVED / "erp_diagnostic.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
