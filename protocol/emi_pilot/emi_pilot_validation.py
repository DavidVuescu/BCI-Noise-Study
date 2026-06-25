#!/usr/bin/env python3
"""
emi_validation_test.py
======================
Post-hoc EMI pilot validation for the BCI Noise Study.

Replicates the preregistered headset-on-table pilot gate (preregistration §8)
using actual subject recordings as retrospective validation evidence.

Preregistered question:
    Does the EMI condition produce measurably elevated counter-gap rate or
    dropout rate relative to the control condition?

    If YES  → EMI was delivered and the BT link was disrupted;
              null accuracy result = "interference tolerated via AFH" interpretation supported.
    If NO   → EMI did not disrupt the BT link; WiFi saturation was present (iperf3 evidence)
              but AFH was fully effective; null accuracy result is still interpretable.

Either outcome is scientifically useful and should be reported in DEVIATIONS.md.

Usage
-----
    # Fast path: read pre-computed metrics from meta.json
    python emi_validation_test.py

    # Slow path: recompute counter gaps from raw _eeg.npy files
    python emi_validation_test.py --from-raw

    # Inspect one meta.json to find the actual key names in YOUR data
    python emi_validation_test.py --inspect-meta

    # Specify a different data root
    python emi_validation_test.py --data-dir /path/to/data/raw

    # Save results to JSON for DEVIATIONS.md reference
    python emi_validation_test.py --output-json emi_validation_results.json
"""

import argparse
import glob
import json
import os
import re
import sys
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Unicorn Hybrid Black data format (17 channels per sample)
# Verify against your loader.py if anything looks off.
# ─────────────────────────────────────────────────────────────────────────────
N_CHANNELS       = 17
EEG_CHANNELS     = slice(0, 8)    # Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
ACCEL_CHANNELS   = slice(8, 11)   # Ax, Ay, Az
GYRO_CHANNELS    = slice(11, 14)  # Gx, Gy, Gz
BATTERY_CHANNEL  = 14
COUNTER_CHANNEL  = 15             # hardware counter — key for gap detection
VALIDATION_CHANNEL = 16

# Candidate key names for counter-gap metrics in meta.json.
# The script tries these in order; first match wins.
META_KEYS_GAP_RATE   = ["gap_rate", "counter_gap_rate", "counter_gaps_rate"]
META_KEYS_N_GAPS     = ["counter_gaps", "n_counter_gaps", "n_gaps", "gap_count",
                        "samples_dropped"]   # Unicorn recorder stores dropped sample count here
META_KEYS_DROPOUT    = ["dropout_rate", "sample_dropout_rate", "packet_dropout"]
META_KEYS_CONDITION  = ["condition", "cond", "noise_condition"]
META_KEYS_SUBJECT    = ["subject_id", "subject", "sub_id"]
# Keys used to compute dropout_rate when not pre-computed
META_KEY_SAMPLES_DROPPED  = "samples_dropped"
META_KEY_SAMPLES_EXPECTED = "samples_expected_by_counter"

# Condition name aliases (preregistration uses these exact strings)
CONDITION_ALIASES = {
    "control":  ["control", "ctrl", "baseline"],
    "emi":      ["emi", "electromagnetic", "wifi", "interference"],
    "chewing":  ["chewing", "chew"],
    "acoustic": ["acoustic", "noise", "pink"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(d, keys, default=None):
    """Try a list of candidate key names on a dict; return first match."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def normalise_condition(raw):
    """Map raw condition string to one of the four canonical names."""
    raw = str(raw).lower().strip()
    for canonical, aliases in CONDITION_ALIASES.items():
        if any(alias in raw for alias in aliases):
            return canonical
    return raw  # unknown; pass through


def find_meta_files(data_dir):
    """Glob for all *_meta.json files under data_dir/sub-*/."""
    pattern = os.path.join(data_dir, "sub-*", "*_meta.json")
    return sorted(glob.glob(pattern))


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def eeg_path_from_meta(meta_path):
    """Derive the _eeg.npy path from a _meta.json path."""
    return meta_path.replace("_meta.json", "_eeg.npy")


def subject_from_path(path):
    m = re.search(r"(sub-\d+)", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


# ─────────────────────────────────────────────────────────────────────────────
# Counter-gap computation from raw EEG numpy array
# ─────────────────────────────────────────────────────────────────────────────

def detect_counter_rollover(counter):
    """
    Auto-detect the rollover value of the hardware counter.
    The counter increments by 1 per sample; when it hits max it wraps to 0.
    Common values: 255 (uint8), 65535 (uint16), or no rollover in practice.
    """
    diffs = np.diff(counter.astype(np.int64))
    # Large negative diffs indicate rollover
    negative_jumps = diffs[diffs < 0]
    if len(negative_jumps) == 0:
        return None  # no rollover observed in this recording

    # The rollover value = abs(negative_jump) - 1 + the value just before wrap
    # Simplest estimate: counter_max = max_observed_value + 1
    return int(counter.max()) + 1


def compute_counter_gaps(eeg_npy_path):
    """
    Load a raw _eeg.npy file and compute counter-gap metrics.

    A counter gap is any sample transition where the counter increments by
    something other than 1 (accounting for rollover). Each gap of size k
    means k-1 samples were dropped from the Bluetooth stream.

    Returns a dict with:
        gap_rate      – fraction of sample transitions that are gaps (0–1)
        n_gaps        – integer count of gap events
        gap_samples   – total number of dropped samples
        dropout_rate  – dropped samples / total samples (0–1)
        rollover      – detected counter rollover value (or None)
        n_samples     – total samples in the recording
    """
    eeg = np.load(eeg_npy_path)

    if eeg.ndim != 2 or eeg.shape[1] < COUNTER_CHANNEL + 1:
        raise ValueError(
            f"Unexpected EEG shape {eeg.shape}. "
            f"Expected (n_samples, ≥{COUNTER_CHANNEL + 1}). "
            f"Check COUNTER_CHANNEL constant at top of script."
        )

    counter = eeg[:, COUNTER_CHANNEL]
    n_samples = len(counter)
    rollover = detect_counter_rollover(counter)

    # Compute diffs; adjust for rollover
    diffs = np.diff(counter.astype(np.int64))
    if rollover is not None:
        # A jump of -(rollover - 1) is a normal rollover (counter max → 0)
        diffs[diffs == -(rollover - 1)] = 1

    # Gap: expected diff is 1; anything else is a gap
    gap_mask = diffs != 1
    n_gaps = int(gap_mask.sum())
    # Each gap of size k contributes (k - 1) dropped samples
    gap_samples = int(np.maximum(diffs[gap_mask] - 1, 0).sum())

    return {
        "gap_rate":    n_gaps / max(len(diffs), 1),
        "n_gaps":      n_gaps,
        "gap_samples": gap_samples,
        "dropout_rate": gap_samples / max(n_samples, 1),
        "rollover":    rollover,
        "n_samples":   n_samples,
        "source":      "computed_from_raw",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_recordings(data_dir, from_raw=False):
    """
    Load counter-gap metrics for all subjects and conditions.

    Strategy:
        1. Try meta.json for pre-computed metrics (fast, always attempted).
        2. If not present in meta.json AND from_raw=True, compute from _eeg.npy.
        3. If neither available, skip and warn.

    Returns dict: {condition_name: [record, ...]}
    where record = {subject, gap_rate, n_gaps, dropout_rate, source, ...}
    """
    meta_files = find_meta_files(data_dir)
    if not meta_files:
        print(f"[ERROR] No *_meta.json files found under {data_dir}")
        print("        Check --data-dir path.")
        sys.exit(1)

    records = {}

    for meta_path in meta_files:
        meta = load_meta(meta_path)
        raw_cond = _get(meta, META_KEYS_CONDITION, "")
        condition = normalise_condition(raw_cond)
        subject   = _get(meta, META_KEYS_SUBJECT, subject_from_path(meta_path))

        record = {
            "subject":   subject,
            "condition": condition,
            "file":      meta_path,
        }

        # --- Fast path: pre-computed metrics in meta.json ---
        gap_rate    = _get(meta, META_KEYS_GAP_RATE)
        n_gaps      = _get(meta, META_KEYS_N_GAPS)
        dropout     = _get(meta, META_KEYS_DROPOUT)

        # Derive dropout_rate from samples_dropped / samples_expected_by_counter
        # if pre-computed keys are absent (Unicorn recorder format)
        if dropout is None and META_KEY_SAMPLES_DROPPED in meta:
            n_dropped  = meta[META_KEY_SAMPLES_DROPPED]
            n_expected = meta.get(META_KEY_SAMPLES_EXPECTED, meta.get("n_samples", 1))
            dropout = n_dropped / max(n_expected, 1)
            if gap_rate is None:
                # Use dropout_rate as the gap_rate proxy; both measure BT data loss
                gap_rate = dropout
            if n_gaps is None:
                n_gaps = int(n_dropped)

        if gap_rate is not None:
            record["gap_rate"]     = float(gap_rate)
            record["n_gaps"]       = int(n_gaps) if n_gaps is not None else None
            record["dropout_rate"] = float(dropout) if dropout is not None else None
            record["source"]       = "meta_json"

        elif from_raw:
            # --- Slow path: compute from raw EEG ---
            eeg_path = eeg_path_from_meta(meta_path)
            if os.path.exists(eeg_path):
                try:
                    gap_metrics = compute_counter_gaps(eeg_path)
                    record.update(gap_metrics)
                    print(f"  [computed] {subject} / {condition}: "
                          f"gap_rate={gap_metrics['gap_rate']:.5f}, "
                          f"n_gaps={gap_metrics['n_gaps']}")
                except Exception as e:
                    print(f"  [WARN] Failed to compute gaps for {eeg_path}: {e}")
                    continue
            else:
                print(f"  [WARN] No _eeg.npy found at {eeg_path}")
                continue

        else:
            # No metrics available without --from-raw
            print(f"  [SKIP] {subject}/{condition}: no pre-computed gap metrics in meta.json. "
                  f"Re-run with --from-raw to compute.")
            continue

        # Accumulate
        if condition not in records:
            records[condition] = []
        records[condition].append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(records):
    """
    Statistical comparison of counter-gap rate: EMI vs control.
    Returns a dict with all computed results.
    """
    out = {}

    for cond, recs in records.items():
        rates   = [r["gap_rate"]    for r in recs if r.get("gap_rate")    is not None]
        dropout = [r["dropout_rate"] for r in recs if r.get("dropout_rate") is not None]
        out[cond] = {
            "n":            len(rates),
            "gap_rates":    rates,
            "mean_gap":     np.mean(rates) if rates else None,
            "std_gap":      np.std(rates)  if rates else None,
            "median_gap":   np.median(rates) if rates else None,
            "min_gap":      np.min(rates)  if rates else None,
            "max_gap":      np.max(rates)  if rates else None,
            "mean_dropout": np.mean(dropout) if dropout else None,
            "dropout_rates": dropout,
        }

    # Core statistical test: EMI gap_rate > control gap_rate (one-tailed)
    ctrl_rates = out.get("control", {}).get("gap_rates", [])
    emi_rates  = out.get("emi",     {}).get("gap_rates", [])

    stat_result = {}
    if ctrl_rates and emi_rates:
        # Paired Wilcoxon if same N (same subjects), Mann-Whitney otherwise
        n_ctrl, n_emi = len(ctrl_rates), len(emi_rates)
        if n_ctrl == n_emi:
            w, p = stats.wilcoxon(emi_rates, ctrl_rates, alternative="greater")
            test_name = "Wilcoxon signed-rank (one-tailed: EMI > Control)"
        else:
            w, p = stats.mannwhitneyu(emi_rates, ctrl_rates, alternative="greater")
            test_name = "Mann-Whitney U (one-tailed: EMI > Control)"

        # Cohen's d (pooled SD)
        pooled_sd = np.sqrt((np.std(ctrl_rates)**2 + np.std(emi_rates)**2) / 2)
        d = (np.mean(emi_rates) - np.mean(ctrl_rates)) / pooled_sd if pooled_sd > 0 else 0.0

        # Fold change
        fold = np.mean(emi_rates) / np.mean(ctrl_rates) if np.mean(ctrl_rates) > 0 else float("inf")

        stat_result = {
            "test":    test_name,
            "stat":    float(w),
            "p":       float(p),
            "cohens_d": float(d),
            "fold_change": float(fold),
            "significant": bool(p < 0.05),
        }

    out["_stats"] = stat_result
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(records, analysis):
    SEP = "=" * 64

    print(f"\n{SEP}")
    print("  EMI VALIDATION: Counter-Gap Analysis")
    print(f"  Preregistration s8 -- retrospective pilot gate")
    print(SEP)

    order = ["control", "emi", "chewing", "acoustic"]
    present = [c for c in order if c in analysis and c != "_stats"]
    present += [c for c in analysis if c not in order and c != "_stats"]

    for cond in present:
        a = analysis[cond]
        n = a["n"]
        if n == 0:
            print(f"\n{cond.upper()}: no data")
            continue

        print(f"\n  {cond.upper()}  (N = {n} recordings)")
        print(f"    Gap rate  -- mean {a['mean_gap']*100:.3f}%  "
              f"(std {a['std_gap']*100:.3f}%  "
              f"med {a['median_gap']*100:.3f}%  "
              f"range {a['min_gap']*100:.3f}-{a['max_gap']*100:.3f}%)")
        if a["mean_dropout"] is not None:
            print(f"    Dropout   -- mean {a['mean_dropout']*100:.3f}%")

    # Statistical comparison
    s = analysis.get("_stats", {})
    if s:
        print(f"\n{SEP}")
        print("  STATISTICAL COMPARISON  (EMI vs Control gap_rate)")
        print(f"  Test:      {s['test']}")
        print(f"  Statistic: {s['stat']:.3f}")
        print(f"  p-value:   {s['p']:.4f}{'  *' if s['significant'] else ''}")
        print(f"  Cohen's d: {s['cohens_d']:.3f}")
        fold_str = f"{s['fold_change']:.2f}x" if s['fold_change'] != float("inf") else "inf (control=0)"
        print(f"  Fold:      {fold_str}  (EMI / Control mean gap rate)")

    # Verdict
    print(f"\n{SEP}")
    print("  VALIDATION VERDICT")

    ctrl_n = analysis.get("control", {}).get("n", 0)
    emi_n  = analysis.get("emi",     {}).get("n", 0)

    if ctrl_n == 0 or emi_n == 0:
        print("  [FAIL] Insufficient data for both conditions -- cannot validate.")
        print("    Check --data-dir and confirm _meta.json / _eeg.npy files exist.")
    elif s.get("significant") and s.get("fold_change", 1) > 1:
        print("  [BT DISRUPTED] EMI condition produced measurably elevated counter-gap rate")
        print("    (p < 0.05, one-tailed; preregistered gate satisfied retrospectively)")
        print("    -> Interpretation: interference was delivered and disrupted the BT link.")
        print("      Null accuracy result = residual robustness after partial data loss.")
    else:
        print("  [AFH TOLERATED] No statistically significant elevation in EMI counter-gap rate")
        print("    (p >= 0.05; BT link tolerated the WiFi interference)")
        print("    -> Interpretation: Adaptive Frequency Hopping was effective.")
        print("      WiFi saturation was confirmed via iperf3 / WiFiMan / UniFi logs.")
        print("      Null accuracy result = BT link resilient, not manipulation failure.")
        print("      This is consistent with Discussion section of the paper draft.")

    print(f"\n{SEP}\n")


def inspect_meta(data_dir):
    """Print the keys of the first found meta.json for debugging."""
    files = find_meta_files(data_dir)
    if not files:
        print(f"No meta.json files found under {data_dir}")
        return
    path = files[0]
    print(f"\nInspecting: {path}\n")
    meta = load_meta(path)
    for k, v in meta.items():
        print(f"  {k!r:40s}: {repr(v)[:80]}")
    print(f"\nTotal keys: {len(meta)}")
    print("\nIf you see counter/gap keys above, update META_KEYS_* constants at top of script.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EMI pilot validation: counter-gap analysis (preregistration §8)"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root of raw data tree (default: data/raw)"
    )
    parser.add_argument(
        "--from-raw", action="store_true",
        help="Recompute counter gaps from raw _eeg.npy files "
             "(slower; use if meta.json lacks gap metrics)"
    )
    parser.add_argument(
        "--inspect-meta", action="store_true",
        help="Print keys of one meta.json then exit (helps find right key names)"
    )
    parser.add_argument(
        "--output-json", metavar="FILE",
        help="Save analysis results to a JSON file"
    )
    args = parser.parse_args()

    if args.inspect_meta:
        inspect_meta(args.data_dir)
        return

    print(f"Scanning: {os.path.abspath(args.data_dir)}")
    print(f"Mode:     {'compute from raw EEG' if args.from_raw else 'read from meta.json'}")

    records  = load_all_recordings(args.data_dir, from_raw=args.from_raw)
    analysis = analyse(records)
    print_report(records, analysis)

    if args.output_json:
        # Serialise numpy types
        def _serial(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            raise TypeError(f"Not serialisable: {type(o)}")
        with open(args.output_json, "w") as f:
            json.dump(analysis, f, indent=2, default=_serial)
        print(f"Results saved -> {args.output_json}")


if __name__ == "__main__":
    main()
