"""Quantitative signal diagnostic for the control recording.

Replaces the visual ERP plot with numbers we can read directly. Asks three
questions:
  1. Where in time is the target-vs-nontarget difference largest? (peak latency)
  2. How strong is it at the expected P300 location (Cz, Pz)? (amplitude + SNR)
  3. Is the spatial pattern coherent with a P300 (central-parietal) or scattered?
"""
from pathlib import Path

import mne
import numpy as np


SUBJECT = "pilot-self-day0"
DERIVED = Path("data/derived/preprocessing-v1") / f"sub-{SUBJECT}"

# Time windows of interest
P300_WIN = (0.250, 0.450)   # s, where P300 should live
N170_WIN = (0.130, 0.200)
BASELINE_WIN = (-0.200, 0.0)


def fmt_uv(v: float) -> str:
    return f"{v*1e6:+7.3f} µV"


def fmt_ratio(r: float) -> str:
    return f"{r:5.2f}"


def diagnose(cond: str):
    path = DERIVED / f"sub-{SUBJECT}_cond-{cond}_epo.fif"
    epochs = mne.read_epochs(path, preload=True, verbose="WARNING")

    target_mask = epochs.metadata["is_target"].values.astype(bool)
    targets = epochs[target_mask]
    nontargets = epochs[~target_mask]
    n_tgt = len(targets)
    n_nt = len(nontargets)

    times = epochs.times  # in seconds
    ch_names = epochs.ch_names

    # Averaged ERPs as ndarrays (n_channels, n_times) in volts
    tgt_avg = targets.average().data
    nt_avg = nontargets.average().data
    diff = tgt_avg - nt_avg  # target - nontarget difference wave

    # Per-trial SD across target epochs (for SNR computation)
    tgt_data = targets.get_data()  # (n_epochs, n_channels, n_times)
    nt_data = nontargets.get_data()
    # Per-channel-per-timepoint std across epochs:
    tgt_std = tgt_data.std(axis=0)
    # Standard error of the mean:
    tgt_sem = tgt_std / np.sqrt(n_tgt)

    print(f"\n{'='*70}")
    print(f"{cond.upper()}  (n_target={n_tgt}, n_nontarget={n_nt})")
    print(f"{'='*70}")

    # ---------- 1. WHERE IS THE LARGEST DIFFERENCE IN TIME? ----------
    # Average absolute difference across central-parietal channels (Cz, Pz)
    # — this is where P300 should live spatially.
    central_channels = ["Cz", "Pz"]
    ch_idxs_cp = [ch_names.index(c) for c in central_channels]
    diff_cp = diff[ch_idxs_cp].mean(axis=0)  # mean across Cz/Pz

    # Window the search to post-stimulus, find peak
    post_mask = times > 0
    post_diff = diff_cp[post_mask]
    post_times = times[post_mask]
    peak_idx = int(np.argmax(np.abs(post_diff)))
    peak_t = post_times[peak_idx] * 1000  # ms
    peak_v = post_diff[peak_idx]

    print(f"\n  TIMING (Cz+Pz mean, target-nontarget difference wave)")
    print(f"  Largest post-stim deflection: {peak_v*1e6:+.3f} µV at {peak_t:.0f} ms")
    if 250 <= peak_t <= 450:
        print(f"    ✓ Peak falls in expected P300 window (250-450 ms)")
    elif 100 <= peak_t < 250:
        print(f"    ⚠ Peak is EARLY (before P300 window).")
        print(f"      May indicate timing misalignment: stimulus marker may")
        print(f"      be delayed vs actual brain response, OR signal is")
        print(f"      dominated by early sensory response not P300.")
    elif 450 < peak_t <= 700:
        print(f"    ⚠ Peak is LATE (after typical P300 window).")
        print(f"      Could indicate slow/delayed P300, or alignment offset")
        print(f"      pushing the response window later than expected.")
    else:
        print(f"    ⚠ Peak is outside any expected ERP window — signal may")
        print(f"      be noise rather than ERP.")

    # ---------- 2. AMPLITUDE & SNR AT EXPECTED P300 LOCATION ----------
    p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])

    print(f"\n  P300 WINDOW ({int(P300_WIN[0]*1000)}-{int(P300_WIN[1]*1000)} ms)")
    print(f"  {'Channel':<8} {'tgt mean':>12} {'nt mean':>12} {'diff':>12} {'SNR':>8}")
    for ch in ["Fz", "C3", "Cz", "C4", "Pz"]:
        ci = ch_names.index(ch)
        tgt_in_window = tgt_avg[ci, p300_mask].mean()
        nt_in_window = nt_avg[ci, p300_mask].mean()
        diff_val = tgt_in_window - nt_in_window
        # SNR: difference amplitude vs std-error of the target mean in window
        sem_in_window = tgt_sem[ci, p300_mask].mean()
        snr = abs(diff_val) / sem_in_window if sem_in_window > 0 else 0.0
        print(f"  {ch:<8} {fmt_uv(tgt_in_window)}  {fmt_uv(nt_in_window)}  "
              f"{fmt_uv(diff_val)}  {fmt_ratio(snr)}")

    print(f"\n    SNR interpretation:")
    print(f"      > 3.0  = robust P300")
    print(f"      1.5-3  = present but small (typical consumer-grade)")
    print(f"      0.5-1.5 = marginal, will struggle to classify")
    print(f"      < 0.5  = effectively absent")

    # ---------- 3. SPATIAL PATTERN AT P300 LATENCY ----------
    # Find the strongest spatial peak in the P300 window across all channels,
    # see whether it's central/parietal (good) or scattered (bad)
    print(f"\n  SPATIAL PATTERN (mean diff in P300 window, all channels)")
    diff_in_window = diff[:, p300_mask].mean(axis=1)
    # Sort channels by absolute difference magnitude
    ranked = sorted(enumerate(diff_in_window), key=lambda x: -abs(x[1]))
    for rank, (ci, v) in enumerate(ranked, start=1):
        marker = ""
        if ch_names[ci] in ("Cz", "Pz", "C3", "C4"):
            marker = " ← central/parietal (P300-relevant)"
        print(f"    {rank}. {ch_names[ci]:<5} {fmt_uv(v)}{marker}")

    # ---------- 4. N170 WINDOW (sanity check for face-processing) ----------
    n170_mask = (times >= N170_WIN[0]) & (times <= N170_WIN[1])
    print(f"\n  N170 WINDOW ({int(N170_WIN[0]*1000)}-{int(N170_WIN[1]*1000)} ms)")
    print(f"  Posterior channels (target-nontarget difference):")
    for ch in ["PO7", "Oz", "PO8"]:
        ci = ch_names.index(ch)
        tgt_in_window = tgt_avg[ci, n170_mask].mean()
        nt_in_window = nt_avg[ci, n170_mask].mean()
        diff_val = tgt_in_window - nt_in_window
        sem_in_window = tgt_sem[ci, n170_mask].mean()
        snr = abs(diff_val) / sem_in_window if sem_in_window > 0 else 0.0
        print(f"    {ch:<5} diff = {fmt_uv(diff_val)}  SNR = {fmt_ratio(snr)}")
    print(f"\n    (N170 fires for any face; difference here reflects whether")
    print(f"    your attention enhanced the face-processing response on targets.)")


def main():
    for cond in ["control", "chewing", "emi", "acoustic"]:
        diagnose(cond)
    print()


if __name__ == "__main__":
    main()
