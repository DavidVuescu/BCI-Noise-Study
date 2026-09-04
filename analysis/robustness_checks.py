"""Robustness of the reported conclusions to marker-alignment quality.

A small number of recordings carry large clock-fit residuals (see
analysis/timing_diagnostic.py), indicating that their stimulus markers are
misaligned with the EEG. This module asks the only question that matters for the
manuscript: do the reported conclusions depend on those recordings?

It answers that by exclusion rather than by correction. Re-aligning affected
recordings would mean adopting a different clock estimator for the whole dataset
and re-deriving every reported value; excluding them and showing the conclusions
hold is the conventional, weaker, and more defensible move. No reported value is
altered by anything in this module.

POST-HOC STATUS
---------------------------------------------------------------------------
The residual thresholds used here were chosen AFTER the residual distribution was
known. They are not a pre-specified rule and are not presented as one. To make
the arbitrariness visible rather than hidden, every analysis is reported at two
cuts -- the three recordings with the largest residuals, and all eleven above a
looser flag -- so a reader can see whether the conclusion depends on where the
line falls. It does not.

CONTENTS
---------------------------------------------------------------------------
    exclusion_sensitivity()      primary contrasts with affected subjects dropped
    n170_exclusion_check()       the same for the secondary N170 analysis
    n170_convergence_check()     the chewing N170 / rejection-rate correlation
    emi_distribution_check()     shape statistics of the EMI accuracy distribution
    below_chance_scan()          recordings ranking targets below non-targets
    registered_gate_recompute()  the N=22 registered-rule contrast, both pipelines
    run_all()                    everything, as one serialisable dict

Outputs land in data/derived/robustness-checks/.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

CLF_CONDITIONS = ["control_heldout", "chewing", "emi", "acoustic"]
RAW_CONDITIONS = ["control", "chewing", "emi", "acoustic"]

DERIVED_PIPELINE = "robustness-checks"
RESIDUAL_SCAN = "timing-diagnostic/residual_scan.csv"

# Residual cuts, in ms. Both are post-hoc; see module docstring. The looser cut
# is the flag used in timing_diagnostic.py, the stricter one isolates the three
# recordings whose misalignment is severe enough to destroy classification.
CUT_SEVERE_MS = 90.0
CUT_FLAGGED_MS = 30.0

# Registered exclusion rule (pre-registration section 4, criterion 3): a subject
# exceeding 20% epoch rejection on ANY condition. Narrowed to control alone in
# the analysis actually run; see DEVIATIONS.md 2026-06-07.
REGISTERED_REJECTION_CEILING = 20.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_accuracy(pipeline: str = "classifier-v3",
                  derived_root: str | Path = "data/derived") -> dict[str, dict]:
    """{subject: {condition: balanced accuracy %}} from a classifier pipeline."""
    out = {}
    pattern = str(Path(derived_root) / pipeline / "sub-*" / "sub-*_results.json")
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        out[d["subject_id"]] = {
            c: d["per_condition"][c]["balanced_accuracy"] * 100
            for c in CLF_CONDITIONS if c in d["per_condition"]
        }
    return out


def load_n170(derived_root: str | Path = "data/derived") -> dict[str, dict]:
    out = {}
    pattern = str(Path(derived_root) / "n170-v1" / "sub-*" / "sub-*_n170.json")
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        out[d["subject_id"]] = {
            c: d["per_condition"][c]["n170_amplitude_uv"] for c in RAW_CONDITIONS
        }
    return out


def load_rejection(derived_root: str | Path = "data/derived") -> dict[str, dict]:
    out: dict[str, dict] = collections.defaultdict(dict)
    pattern = str(Path(derived_root) / "preprocessing-v2" / "sub-*"
                  / "sub-*_cond-*_rejection.json")
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        out[d["subject_id"]][d["condition"]] = d["rejection_rate"] * 100
    return dict(out)


def subjects_above_residual(threshold_ms: float,
                            derived_root: str | Path = "data/derived") -> list[str]:
    """Subjects owning at least one recording above a clock-fit residual cut.

    Exclusion operates at subject level because the design is within-subject and
    the group tests use list-wise deletion; dropping a single condition would
    leave an incomplete subject that the tests would discard anyway.
    """
    path = Path(derived_root) / RESIDUAL_SCAN
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/timing_diagnostic.ipynb section 2 "
            f"(or analysis.timing_diagnostic.scan_all_recordings) first.")
    scan = pd.read_csv(path, dtype={"subject": str})
    return sorted(scan.loc[scan.residual_std_ms > threshold_ms, "subject"].unique())


# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------

def _paired_dz(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd else 0.0


def _contrast_block(matrix: np.ndarray, conditions: list[str],
                    alternative: str) -> dict:
    """Friedman omnibus plus Holm-corrected one-tailed Wilcoxon vs the first column.

    Mirrors the procedure in analysis/classifier.py and analysis/n170.py so that
    results here are directly comparable to the reported ones.
    """
    chi2, p_omni = friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    raw, effects = [], {}
    for j, cond in enumerate(conditions[1:], start=1):
        _, p = wilcoxon(matrix[:, j], matrix[:, 0], alternative=alternative)
        raw.append(float(p))
        effects[cond] = {"mean": float(matrix[:, j].mean()),
                         "dz": _paired_dz(matrix[:, j], matrix[:, 0]),
                         "p_one_tailed": float(p)}
    holm = multipletests(raw, alpha=0.05, method="holm")[1]
    for cond, p_adj in zip(effects, holm):
        effects[cond]["p_holm"] = float(p_adj)
        effects[cond]["significant_holm"] = bool(p_adj < 0.05)
    return {
        "n": int(matrix.shape[0]),
        "reference_mean": float(matrix[:, 0].mean()),
        "friedman": {"chi2": float(chi2), "p": float(p_omni)},
        "contrasts": effects,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def exclusion_sensitivity(pipeline: str = "classifier-v3",
                          derived_root: str | Path = "data/derived") -> dict:
    """Primary balanced-accuracy contrasts with alignment-affected subjects dropped.

    Reported at both residual cuts. `alternative="less"` matches the registered
    direction (noise degrades accuracy).
    """
    acc = load_accuracy(pipeline, derived_root)
    subjects = sorted(acc)

    variants = {
        "full": [],
        f"drop_residual_gt_{int(CUT_SEVERE_MS)}ms":
            subjects_above_residual(CUT_SEVERE_MS, derived_root),
        f"drop_residual_gt_{int(CUT_FLAGGED_MS)}ms":
            subjects_above_residual(CUT_FLAGGED_MS, derived_root),
    }

    out = {"pipeline": pipeline, "variants": {}}
    for label, dropped in variants.items():
        kept = [s for s in subjects if s not in set(dropped)]
        M = np.array([[acc[s][c] for c in CLF_CONDITIONS] for s in kept])
        block = _contrast_block(M, CLF_CONDITIONS, alternative="less")
        block["excluded_subjects"] = list(dropped)
        out["variants"][label] = block
    return out


def n170_exclusion_check(derived_root: str | Path = "data/derived") -> dict:
    """Secondary N170 contrasts with alignment-affected subjects dropped.

    `alternative="greater"`: the registered direction is that noise makes the
    posterior amplitude less negative.
    """
    n170 = load_n170(derived_root)
    subjects = sorted(n170)
    variants = {
        "full": [],
        f"drop_residual_gt_{int(CUT_SEVERE_MS)}ms":
            subjects_above_residual(CUT_SEVERE_MS, derived_root),
    }
    out = {"variants": {}}
    for label, dropped in variants.items():
        kept = [s for s in subjects if s not in set(dropped)]
        M = np.array([[n170[s][c] for c in RAW_CONDITIONS] for s in kept])
        block = _contrast_block(M, RAW_CONDITIONS, alternative="greater")
        block["excluded_subjects"] = list(dropped)
        out["variants"][label] = block
    return out


def n170_convergence_check(derived_root: str | Path = "data/derived") -> dict:
    """Correlation between chewing posterior amplitude and chewing rejection rate.

    Two definitions of the dependent measure are reported because they differ:
    the raw chewing amplitude, and the shift from that subject's own control. The
    reported manuscript value corresponds to the RAW amplitude. Both are given so
    that the reported figure is not accidentally replaced by the other one.
    """
    n170, rej = load_n170(derived_root), load_rejection(derived_root)
    subjects = sorted(set(n170) & set(rej))

    def _corr(kept: list[str]) -> dict:
        y = np.array([rej[s]["chewing"] for s in kept])
        raw = np.array([n170[s]["chewing"] for s in kept])
        shift = np.array([n170[s]["chewing"] - n170[s]["control"] for s in kept])
        res = {"n": len(kept)}
        for name, x in (("raw_amplitude", raw), ("shift_from_control", shift)):
            r, p = st.pearsonr(y, x)
            rho, p_rho = st.spearmanr(y, x)
            res[name] = {"pearson_r": float(r), "pearson_p": float(p),
                         "spearman_rho": float(rho), "spearman_p": float(p_rho)}
        return res

    out = {"full": _corr(subjects)}
    for cut in (CUT_SEVERE_MS, CUT_FLAGGED_MS):
        dropped = set(subjects_above_residual(cut, derived_root))
        out[f"drop_residual_gt_{int(cut)}ms"] = _corr(
            [s for s in subjects if s not in dropped])
    return out


def emi_distribution_check(pipeline: str = "classifier-v3",
                           derived_root: str | Path = "data/derived",
                           seed: int = 20260526, n_boot: int = 10000) -> dict:
    """Shape statistics of the per-subject EMI accuracy distribution.

    The manuscript characterises this distribution as having a heavy lower tail.
    Recomputing it without the alignment-affected subjects shows how much of that
    shape those recordings account for.
    """
    acc = load_accuracy(pipeline, derived_root)
    subjects = sorted(acc)

    def _shape(kept: list[str]) -> dict:
        v = np.array([acc[s]["emi"] for s in kept])
        n = len(v)
        g = float(st.skew(v, bias=False))
        k = float(st.kurtosis(v, fisher=True, bias=False))
        bc = (g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        res = {"n": n, "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
               "skewness": g, "excess_kurtosis": k,
               "bimodality_coefficient": float(bc),
               "bimodality_threshold": 0.555,
               "exceeds_bimodality_threshold": bool(bc > 0.555)}
        try:
            import diptest
            d, p = diptest.diptest(v, boot_pval=True, n_boot=n_boot, seed=seed)
            res["hartigan_dip"] = {"D": float(d), "p_bootstrap": float(p)}
        except ImportError:
            res["hartigan_dip"] = None
        return res

    out = {"full": _shape(subjects)}
    for cut in (CUT_SEVERE_MS, CUT_FLAGGED_MS):
        dropped = set(subjects_above_residual(cut, derived_root))
        out[f"drop_residual_gt_{int(cut)}ms"] = _shape(
            [s for s in subjects if s not in dropped])
    return out


def below_chance_scan(derived_root: str | Path = "data/derived") -> dict:
    """Recordings whose classifier ranks targets BELOW non-targets (AUC < 0.5).

    Additive noise degrades AUC toward 0.5 and cannot pass it, so a below-chance
    ranking indicates a systematic error rather than a noisy recording. This makes
    AUC an independent detector of misalignment, complementary to the clock-fit
    residual, and one that requires no timing information at all.

    Requires the sweep in analysis/run_timing_sweep.py to have been run.
    """
    path = Path(derived_root) / "timing-diagnostic" / "full_sweep.csv"
    if not path.exists():
        return {"available": False,
                "note": "run: python -m analysis.run_timing_sweep --workers 6"}
    sweep = pd.read_csv(path, dtype={"subject": str})
    cols = ["subject", "condition", "auc_ols", "residual_std_ms", "flagged"]
    out = {"available": True, "n_recordings": int(len(sweep))}
    for label, cut in (("below_0.50", 0.50), ("below_0.60", 0.60)):
        sel = sweep[sweep.auc_ols < cut].sort_values("auc_ols")
        out[label] = {"n": int(len(sel)), "recordings": sel[cols].to_dict("records")}
    return out


def registered_gate_recompute(derived_root: str | Path = "data/derived") -> dict:
    """Chewing contrast on the subjects the REGISTERED exclusion rule would retain.

    The registered rule excluded any subject exceeding 20% epoch rejection on any
    condition; the analysis actually run narrowed this to control alone
    (DEVIATIONS.md 2026-06-07). The manuscript reports this comparison for the
    published pipeline. It is recomputed here for the blockwise pipeline as well.
    """
    rej = load_rejection(derived_root)
    out = {"rejection_ceiling_pct": REGISTERED_REJECTION_CEILING, "pipelines": {}}
    for pipeline in ("classifier-v3", "classifier-v3-blockwise"):
        acc = load_accuracy(pipeline, derived_root)
        if not acc:
            continue
        kept = [s for s in sorted(acc)
                if s in rej
                and max(rej[s].values()) <= REGISTERED_REJECTION_CEILING]
        if len(kept) < 3:
            continue
        M = np.array([[acc[s]["control_heldout"], acc[s]["chewing"]] for s in kept])
        _, p = wilcoxon(M[:, 1], M[:, 0], alternative="less")
        out["pipelines"][pipeline] = {
            "n": len(kept),
            "control_mean": float(M[:, 0].mean()),
            "chewing_mean": float(M[:, 1].mean()),
            "dz": _paired_dz(M[:, 1], M[:, 0]),
            "p_one_tailed_uncorrected": float(p),
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(derived_root: str | Path = "data/derived") -> dict:
    """Every check, as one serialisable dict."""
    return {
        "residual_cuts_ms": {"severe": CUT_SEVERE_MS, "flagged": CUT_FLAGGED_MS},
        "cuts_are_post_hoc": True,
        "primary_accuracy": exclusion_sensitivity("classifier-v3", derived_root),
        "primary_accuracy_blockwise": exclusion_sensitivity(
            "classifier-v3-blockwise", derived_root),
        "n170": n170_exclusion_check(derived_root),
        "n170_convergence": n170_convergence_check(derived_root),
        "emi_distribution": emi_distribution_check("classifier-v3", derived_root),
        "below_chance": below_chance_scan(derived_root),
        "registered_gate": registered_gate_recompute(derived_root),
    }


def save_summary(results: dict, derived_root: str | Path = "data/derived") -> Path:
    out_dir = Path(derived_root) / DERIVED_PIPELINE
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "robustness_summary.json"
    path.write_text(json.dumps(results, indent=2))
    return path
