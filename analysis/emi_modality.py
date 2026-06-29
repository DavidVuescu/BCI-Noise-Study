"""Unimodality assessment of the per-subject EMI balanced-accuracy distribution.

Tests whether the EMI accuracy distribution departs from unimodality (Hartigan's
dip test) and reports supporting descriptive statistics. The EMI vector is read
from the canonical classifier-v3 result JSONs and checked against the reported
Table I values before analysis.

Dependencies: numpy, scipy, diptest (no mne/sklearn).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import diptest

CLASSIFIER_VERSION = "classifier-v3"
EMI_KEY = "emi"

# Reported Table I values; guard fails if the loaded vector does not match.
REPORTED_EMI_MEAN = 86.9
REPORTED_EMI_SD = 9.8
GUARD_TOL_MEAN = 0.1
GUARD_TOL_SD = 0.1

# Descriptive near-chance cutoff used only to label the low outliers.
# Chance for balanced accuracy is 50%; 60% is a round near-chance band.
# Not a statistical criterion and not the pre-registered control-ceiling floor.
NEAR_CHANCE_CUTOFF = 60.0

DEFAULT_SEED = 20260526
DEFAULT_N_BOOT = 10000
BC_THRESHOLD = 0.555


def load_emi_accuracy(
    derived_root: str | Path = "data/derived",
    version: str = CLASSIFIER_VERSION,
) -> tuple[list[str], np.ndarray, list[Path]]:
    """Return (subject_ids, EMI balanced accuracy %, source files), sorted by id."""
    root = Path(derived_root) / version
    if not root.is_dir():
        raise FileNotFoundError(f"results dir not found: {root}")
    files = sorted(root.glob("sub-*/sub-*_results.json"))
    if not files:
        raise FileNotFoundError(f"no result JSONs under {root}")

    subject_ids: list[str] = []
    acc: list[float] = []
    used: list[Path] = []
    for f in files:
        d = json.loads(f.read_text())
        pc = d.get("per_condition", {})
        if EMI_KEY not in pc:
            continue
        subject_ids.append(str(d["subject_id"]))
        acc.append(pc[EMI_KEY]["balanced_accuracy"] * 100.0)
        used.append(f)

    order = np.argsort(subject_ids)
    subject_ids = [subject_ids[i] for i in order]
    emi = np.asarray([acc[i] for i in order], dtype=float)
    used = [used[i] for i in order]
    return subject_ids, emi, used


def verify_against_reported(
    emi: np.ndarray,
    mean: float = REPORTED_EMI_MEAN,
    sd: float = REPORTED_EMI_SD,
    tol_mean: float = GUARD_TOL_MEAN,
    tol_sd: float = GUARD_TOL_SD,
) -> dict:
    """Raise if the loaded vector does not reproduce the reported Table I values."""
    got_mean = float(np.mean(emi))
    got_sd = float(np.std(emi, ddof=1))
    dm, ds = abs(got_mean - mean), abs(got_sd - sd)
    if dm > tol_mean or ds > tol_sd:
        raise AssertionError(
            "EMI vector does not reproduce reported Table I values.\n"
            f"  mean: got {got_mean:.3f}, expected {mean} (|d|={dm:.3f} > {tol_mean})\n"
            f"  SD:   got {got_sd:.3f}, expected {sd} (|d|={ds:.3f} > {tol_sd})"
        )
    return {
        "passed": True,
        "loaded_mean": round(got_mean, 4),
        "loaded_sd": round(got_sd, 4),
        "reported_mean": mean,
        "reported_sd": sd,
    }


def hartigan_dip(
    emi: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    """Hartigan & Hartigan dip test. Table p (deterministic) and bootstrap p."""
    dip_stat, p_table = diptest.diptest(emi, boot_pval=False)
    _, p_boot = diptest.diptest(emi, boot_pval=True, n_boot=n_boot, seed=seed)
    return {
        "dip_statistic": round(float(dip_stat), 6),
        "p_table": round(float(p_table), 6),
        "p_bootstrap": round(float(p_boot), 6),
        "n_boot": n_boot,
        "seed": seed,
        "rejects_unimodality_at_0.05": bool(p_table < 0.05),
    }


def bimodality_coefficient(emi: np.ndarray) -> dict:
    """Sarle's bimodality coefficient with its skewness/kurtosis inputs."""
    n = len(emi)
    g = float(stats.skew(emi, bias=False))
    k = float(stats.kurtosis(emi, fisher=True, bias=False))
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    bc = (g ** 2 + 1.0) / denom
    return {
        "bimodality_coefficient": round(bc, 4),
        "threshold": BC_THRESHOLD,
        "exceeds_threshold": bool(bc > BC_THRESHOLD),
        "skewness": round(g, 4),
        "excess_kurtosis": round(k, 4),
    }


def collapse_and_normality(
    subject_ids: list[str],
    emi: np.ndarray,
    cutoff: float = NEAR_CHANCE_CUTOFF,
) -> dict:
    """Low outliers below the near-chance cutoff, gap to the bulk, and Shapiro-Wilk
    normality of the full vector and of the bulk with outliers removed."""
    order = np.argsort(emi)
    sorted_emi = emi[order]
    sorted_ids = [subject_ids[i] for i in order]

    low_mask = emi < cutoff
    low = [
        {"subject_id": subject_ids[i], "balanced_accuracy": round(float(emi[i]), 2)}
        for i in np.where(low_mask)[0]
    ]
    n_low = int(low_mask.sum())
    bulk = sorted_emi[n_low:]
    gap = (
        round(float(sorted_emi[n_low] - sorted_emi[n_low - 1]), 2)
        if 0 < n_low < len(sorted_emi) else None
    )
    sw_all = stats.shapiro(emi)
    sw_bulk = stats.shapiro(bulk) if len(bulk) >= 3 else None

    return {
        "near_chance_cutoff": cutoff,
        "n_total": int(len(emi)),
        "n_below_cutoff": n_low,
        "subjects_below_cutoff": low,
        "lowest_in_bulk": round(float(sorted_emi[n_low]), 2) if n_low < len(sorted_emi) else None,
        "gap_to_bulk_pct": gap,
        "bulk_mean": round(float(np.mean(bulk)), 2) if len(bulk) else None,
        "bulk_sd": round(float(np.std(bulk, ddof=1)), 2) if len(bulk) > 1 else None,
        "shapiro_all": {"W": round(float(sw_all.statistic), 4), "p": round(float(sw_all.pvalue), 6)},
        "shapiro_bulk": {"W": round(float(sw_bulk.statistic), 4), "p": round(float(sw_bulk.pvalue), 6)}
        if sw_bulk is not None else None,
        "sorted_low_tail": [
            {"subject_id": sid, "balanced_accuracy": round(float(v), 2)}
            for sid, v in zip(sorted_ids[:6], sorted_emi[:6])
        ],
    }


def _vector_hash(subject_ids: list[str], emi: np.ndarray) -> str:
    payload = json.dumps([[sid, round(float(v), 10)] for sid, v in zip(subject_ids, emi)])
    return hashlib.sha256(payload.encode()).hexdigest()


def provenance(subject_ids: list[str], emi: np.ndarray, source_files: list[Path]) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_subjects": int(len(emi)),
        "classifier_version": CLASSIFIER_VERSION,
        "vector_sha256": _vector_hash(subject_ids, emi),
        "n_source_files": len(source_files),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "diptest": diptest.__version__,
        },
    }


def analyze_emi(
    emi: np.ndarray,
    subject_ids: list[str],
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    source_files: list[Path] | None = None,
    run_guard: bool = True,
) -> dict:
    """Full analysis on an EMI balanced-accuracy vector (%)."""
    emi = np.asarray(emi, dtype=float)
    guard = verify_against_reported(emi) if run_guard else {"passed": None}
    return {
        "summary": {
            "n": int(len(emi)),
            "mean": round(float(np.mean(emi)), 3),
            "sd": round(float(np.std(emi, ddof=1)), 3),
            "min": round(float(np.min(emi)), 3),
            "max": round(float(np.max(emi)), 3),
        },
        "guard": guard,
        "dip_test": hartigan_dip(emi, seed=seed, n_boot=n_boot),
        "bimodality_coefficient": bimodality_coefficient(emi),
        "collapse_and_normality": collapse_and_normality(subject_ids, emi),
        "provenance": provenance(subject_ids, emi, source_files or []),
    }


def run_analysis(
    derived_root: str | Path = "data/derived",
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    subject_ids, emi, files = load_emi_accuracy(derived_root)
    return analyze_emi(emi, subject_ids, seed=seed, n_boot=n_boot, source_files=files)


def save_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return path


def plot_distribution(
    subject_ids: list[str],
    emi: np.ndarray,
    results: dict,
    ax=None,
    save_path: str | Path | None = None,
):
    """Histogram and rug of the EMI balanced-accuracy distribution."""
    import matplotlib.pyplot as plt

    emi = np.asarray(emi, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
    else:
        fig = ax.figure

    cutoff = results["collapse_and_normality"]["near_chance_cutoff"]
    bins = np.arange(45, 102.5, 2.5)
    ax.hist(emi, bins=bins, color="#bcd", edgecolor="white", alpha=0.9, label="subjects")
    ax.plot(emi, np.full_like(emi, -0.4), "|", color="#445", markersize=9, alpha=0.7)
    ax.axvline(cutoff, color="#c33", ls="--", lw=1.2, label=f"near-chance cutoff ({cutoff:.0f}%)")

    dip = results["dip_test"]
    ax.set_title(f"EMI balanced accuracy (N={len(emi)}); Hartigan dip p={dip['p_table']:.2f}", fontsize=10)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_ylabel("Count")
    ax.set_xlim(45, 100)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EMI accuracy unimodality assessment")
    ap.add_argument("--derived-root", default="data/derived")
    ap.add_argument("--out", default="data/derived/emi-bimodality/emi_bimodality_results.json")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = ap.parse_args(argv)

    subject_ids, emi, files = load_emi_accuracy(args.derived_root)
    results = analyze_emi(emi, subject_ids, seed=args.seed, n_boot=args.n_boot, source_files=files)
    out = save_results(results, args.out)
    if args.figure:
        plot_distribution(subject_ids, emi, results, save_path=args.figure)

    s, d = results["summary"], results["dip_test"]
    print(f"N={s['n']}  mean={s['mean']}  SD={s['sd']}  min={s['min']}  max={s['max']}")
    print(f"dip D={d['dip_statistic']}  p_table={d['p_table']}  p_boot={d['p_bootstrap']}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
