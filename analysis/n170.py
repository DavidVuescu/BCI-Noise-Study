"""
N170 secondary analysis: face-evoked negative deflection at posterior channels.

Pre-registered as the secondary confirmatory analysis (prerec.md §5).
Measured as mean voltage in the 130–200 ms window at PO7, Oz, PO8 across
ALL clean face-flash epochs (target + nontarget), per condition per subject.
N170 is a face-processing response not contingent on attention — using all
epochs rather than target-only recovers statistical power without retinotopic
confound (prerec §5).

Output paths:
    data/derived/n170-v1/sub-<id>/sub-<id>_n170.json
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, t as t_dist
from statsmodels.stats.anova import AnovaRM


# ---- Pre-registered parameters -------------------------------------------
N170_CHANNELS = ["PO7", "Oz", "PO8"]   # posterior channels; indices 5,6,7 in Unicorn layout
N170_TMIN_S = 0.130                      # 130 ms post-flash
N170_TMAX_S = 0.200                      # 200 ms post-flash

DERIVED_PIPELINE = "n170-v1"
PREPROCESSING_PIPELINE = "preprocessing-v2"

CONDITIONS = ["control", "chewing", "emi", "acoustic"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_epochs(subject_id: str, condition: str, derived_root: Path) -> mne.Epochs:
    path = (derived_root / PREPROCESSING_PIPELINE / f"sub-{subject_id}"
            / f"sub-{subject_id}_cond-{condition}_epo.fif")
    if not path.exists():
        raise FileNotFoundError(
            f"No preprocessed epochs at {path}. Run preprocess_recording first."
        )
    return mne.read_epochs(path, preload=True, verbose="WARNING")


# --------------------------------------------------------------------------
# Per-epoch / per-condition computation
# --------------------------------------------------------------------------

def compute_n170_amplitude(epochs: mne.Epochs) -> float:
    """Mean amplitude (µV) in the N170 window, grand-averaged over all epochs.

    Grand-averages the ERP across all epochs (target + nontarget), picks the
    three posterior channels, restricts to the 130–200 ms window, and returns
    the mean over channels × time.

    A present N170 yields a negative return value; noise attenuating the N170
    makes the return value less negative (closer to zero or positive).
    """
    time_mask = (epochs.times >= N170_TMIN_S) & (epochs.times <= N170_TMAX_S)
    evoked = epochs.average()
    picks = [evoked.ch_names.index(ch) for ch in N170_CHANNELS]
    data_uv = evoked.data[picks][:, time_mask] * 1e6  # volts → µV
    return float(data_uv.mean())


def extract_evoked_waveforms(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Grand-average ERP waveforms at the three N170 channels.

    Returns:
        times_s: (n_times,) array of time points in seconds
        waveforms: dict mapping each channel name → amplitude list in µV
    """
    evoked = epochs.average()
    times_s = evoked.times.copy()
    waveforms: dict[str, list[float]] = {}
    for ch in N170_CHANNELS:
        idx = evoked.ch_names.index(ch)
        waveforms[ch] = (evoked.data[idx] * 1e6).tolist()
    return times_s, waveforms


# --------------------------------------------------------------------------
# Subject-level pipeline
# --------------------------------------------------------------------------

def run_n170_subject(
    subject_id: str,
    derived_root: str | Path = "data/derived",
    save: bool = True,
) -> dict:
    """Compute N170 amplitude for all four conditions for one subject.

    Reads preprocessed epochs from data/derived/preprocessing-v2/.
    All clean epochs (target + nontarget) are used per prerec §5.

    Returns a dict suitable for JSON serialization and group-level loading.
    """
    derived_root = Path(derived_root)
    per_condition: dict = {}

    for cond in CONDITIONS:
        epochs = _load_epochs(subject_id, cond, derived_root)
        amp = compute_n170_amplitude(epochs)
        times_s, waveforms = extract_evoked_waveforms(epochs)
        per_condition[cond] = {
            "n170_amplitude_uv": amp,
            "n_epochs": len(epochs),
            "n_target_epochs": int((epochs.metadata["is_target"].astype(bool)).sum()),
            "n_nontarget_epochs": int((~epochs.metadata["is_target"].astype(bool)).sum()),
            "evoked_times_s": times_s.tolist(),
            "evoked_per_channel_uv": waveforms,
        }

    results = {
        "subject_id": subject_id,
        "parameters": {
            "n170_channels": N170_CHANNELS,
            "window_tmin_s": N170_TMIN_S,
            "window_tmax_s": N170_TMAX_S,
            "preprocessing_pipeline": PREPROCESSING_PIPELINE,
        },
        "per_condition": per_condition,
    }

    if save:
        out_dir = derived_root / DERIVED_PIPELINE / f"sub-{subject_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sub-{subject_id}_n170.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# --------------------------------------------------------------------------
# Group loading
# --------------------------------------------------------------------------

def load_all_n170_results(derived_root: str | Path = "data/derived") -> list[dict]:
    """Scan data/derived/n170-v1 and load every _n170.json found."""
    derived_root = Path(derived_root)
    n170_dir = derived_root / DERIVED_PIPELINE
    results = []
    for path in sorted(n170_dir.glob("sub-*/sub-*_n170.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def build_amplitude_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a tidy long-format DataFrame from a list of n170 result dicts.

    Returns a DataFrame with columns: subject, condition, n170_amplitude_uv,
    n_epochs.
    """
    rows = []
    for r in results:
        subj = r["subject_id"]
        for cond in CONDITIONS:
            if cond not in r["per_condition"]:
                continue
            pc = r["per_condition"][cond]
            rows.append({
                "subject": subj,
                "condition": cond,
                "n170_amplitude_uv": pc["n170_amplitude_uv"],
                "n_epochs": pc["n_epochs"],
            })
    return pd.DataFrame(rows)


def _build_amplitude_matrix(
    results: list[dict],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build (n_subjects, n_conditions) amplitude matrix for statistics.

    Returns:
        matrix: (n, k) array in µV; NaN for missing conditions
        subjects: ordered list of subject IDs
        conditions: ordered list of condition names
    """
    subjects = [r["subject_id"] for r in results]
    n, k = len(subjects), len(CONDITIONS)
    matrix = np.full((n, k), np.nan)
    for i, r in enumerate(results):
        for j, cond in enumerate(CONDITIONS):
            if cond in r["per_condition"]:
                matrix[i, j] = r["per_condition"][cond]["n170_amplitude_uv"]
    return matrix, subjects, list(CONDITIONS)


# --------------------------------------------------------------------------
# Group-level statistics
# --------------------------------------------------------------------------

def _greenhouse_geisser_epsilon(data: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon, computed in the (k-1)-dim contrast space.

    data: (n_subjects, n_conditions). The condition covariance is projected onto
    an orthonormal contrast basis (which removes the shared 'all-conditions'
    axis) BEFORE the trace ratio. The earlier implementation applied the ratio
    to the RAW covariance, leaving that common axis in; because subjects are
    positively correlated across conditions the common axis dominates, which
    pinned epsilon near the 1/(k-1) floor and over-corrected. Clipped to
    [1/(k-1), 1].
    """
    k = data.shape[1]
    S = np.cov(data.T)                           # (k, k) condition covariance
    H = np.eye(k) - np.ones((k, k)) / k          # centering matrix
    u, _s, _vt = np.linalg.svd(H)
    C = u[:, :k - 1].T                           # orthonormal rows, C @ ones = 0
    Sstar = C @ S @ C.T                          # covariance in contrast space
    tr = np.trace(Sstar)
    tr2 = np.trace(Sstar @ Sstar)
    if tr2 <= 0:
        return 1.0
    eps = (tr ** 2) / ((k - 1) * tr2)
    return float(np.clip(eps, 1.0 / (k - 1), 1.0))


def _paired_cohens_d_with_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> dict:
    """Cohen's d for paired samples with analytical 95 % CI.

    d_z = mean(x - y) / std(x - y). CI via the t approximation:
    SE(d_z) ≈ sqrt(1/n + d_z²/(2n)).
    """
    diffs = x - y
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    if sd_d == 0:
        return {"cohens_d": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "mean_diff_uv": mean_d, "sd_diff_uv": sd_d, "n": n}
    d = mean_d / sd_d
    se_d = float(np.sqrt(1.0 / n + d ** 2 / (2.0 * n)))
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    return {
        "cohens_d": float(d),
        "ci_low": float(d - t_crit * se_d),
        "ci_high": float(d + t_crit * se_d),
        "mean_diff_uv": mean_d,
        "sd_diff_uv": sd_d,
        "n": n,
    }


def run_group_statistics(results: list[dict]) -> dict:
    """Pre-registered group-level statistics on N170 amplitudes.

    1. Non-parametric (primary): Friedman test across conditions, then
       pairwise Wilcoxon signed-rank vs control (one-tailed: noise makes
       amplitude less negative, i.e. noise_vals > ctrl_vals).
    2. Parametric (robustness check): repeated-measures ANOVA with
       Greenhouse-Geisser correction.

    Subjects with any missing condition are excluded list-wise.

    Returns:
        Nested dict of results, ready for display or JSON export.
    """
    matrix, subjects, conditions = _build_amplitude_matrix(results)

    # List-wise deletion of subjects with missing conditions
    valid = ~np.any(np.isnan(matrix), axis=1)
    matrix = matrix[valid]
    subjects_valid = [s for s, v in zip(subjects, valid) if v]
    n = len(subjects_valid)

    ctrl_idx = conditions.index("control")
    ctrl_vals = matrix[:, ctrl_idx]

    # ---- Descriptives ----------------------------------------------------
    descriptives: dict = {}
    for j, cond in enumerate(conditions):
        vals = matrix[:, j]
        descriptives[cond] = {
            "n": n,
            "mean_uv": float(vals.mean()),
            "sd_uv": float(vals.std(ddof=1)),
            "sem_uv": float(vals.std(ddof=1) / np.sqrt(n)),
            "median_uv": float(np.median(vals)),
        }

    # ---- 1. Friedman test ------------------------------------------------
    friedman_stat, friedman_p = friedmanchisquare(
        *[matrix[:, j] for j in range(len(conditions))]
    )

    # ---- 2. Wilcoxon post-hoc vs control ---------------------------------
    # Direction: noise amplitude > control amplitude (less negative → 'greater')
    posthoc: dict = {}
    noise_conds = [c for c in conditions if c != "control"]
    for cond in noise_conds:
        j = conditions.index(cond)
        noise_vals = matrix[:, j]
        try:
            stat_two, p_two = wilcoxon(noise_vals, ctrl_vals, alternative="two-sided")
            stat_one, p_one = wilcoxon(noise_vals, ctrl_vals, alternative="greater")
        except Exception:
            stat_two, p_two, stat_one, p_one = np.nan, np.nan, np.nan, np.nan

        effect = _paired_cohens_d_with_ci(noise_vals, ctrl_vals)
        posthoc[cond] = {
            "wilcoxon_statistic": float(stat_two),
            "p_two_tailed": float(p_two),
            "p_one_tailed": float(p_one),
            **effect,
        }

    # ---- 3. Parametric RM-ANOVA with GG correction -----------------------
    rows_long = [
        {"subject": subj, "condition": cond, "n170": float(matrix[i, j])}
        for i, subj in enumerate(subjects_valid)
        for j, cond in enumerate(conditions)
    ]
    df_long = pd.DataFrame(rows_long)

    rm_anova: dict = {}
    try:
        aovrm = AnovaRM(df_long, depvar="n170", subject="subject", within=["condition"])
        fit = aovrm.fit()
        f_stat = float(fit.anova_table["F Value"].iloc[0])
        df_num = float(fit.anova_table["Num DF"].iloc[0])
        df_den = float(fit.anova_table["Den DF"].iloc[0])
        p_uncorr = float(fit.anova_table["Pr > F"].iloc[0])

        gg_eps = _greenhouse_geisser_epsilon(matrix)
        from scipy.stats import f as f_dist
        p_gg = float(1 - f_dist.cdf(f_stat, df_num * gg_eps, df_den * gg_eps))

        rm_anova = {
            "f_statistic": f_stat,
            "df_numerator": df_num,
            "df_denominator": df_den,
            "p_uncorrected": p_uncorr,
            "greenhouse_geisser_epsilon": gg_eps,
            "df_numerator_gg": df_num * gg_eps,
            "df_denominator_gg": df_den * gg_eps,
            "p_greenhouse_geisser": p_gg,
        }
    except Exception as exc:
        rm_anova["error"] = str(exc)

    return {
        "n_subjects": n,
        "subjects_included": subjects_valid,
        "conditions": conditions,
        "descriptives": descriptives,
        "friedman": {
            "statistic": float(friedman_stat),
            "p_value": float(friedman_p),
            "df": len(conditions) - 1,
        },
        "wilcoxon_posthoc_vs_control": posthoc,
        "rm_anova": rm_anova,
    }
