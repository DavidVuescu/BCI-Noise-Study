"""Marker-alignment timing diagnostic: one-sided Bluetooth packet delay.

WHY THIS EXISTS
---------------------------------------------------------------------------
SYNASC 2026 Reviewer 1 objected that our reading of the two near-chance EMI
recordings as "catastrophic failure" caused by RF energy coupling into the
sensing chain "lacks supporting impedance or RF measurements." That objection is
correct: we have no such measurements. This module tests a different and testable
explanation, and finds it.

THE MECHANISM
---------------------------------------------------------------------------
analysis/loader.py aligns stimulus markers to the EEG sample axis by fitting

    acq_time ~ m * (wall_clock_receive_time - anchor) + c

by ordinary least squares over every received UDP packet. OLS is the right
estimator when the noise is symmetric. Bluetooth packet delay is NOT symmetric:
a packet can arrive late, never early. Delay is therefore one-sided and, under
link contention, heavy-tailed.

In acq-vs-arrival space a delayed packet sits BELOW the true line (its arrival
time is inflated relative to its acquisition time). The true clock is therefore
the UPPER ENVELOPE of the point cloud, not its least-squares mean. When the
delay distribution is tight -- which it is in 157 of our 168 recordings, median
residual ~6 ms -- the difference is invisible. When the tail blows out, OLS tilts
toward the delayed mass, and every stimulus marker slides off the EEG.

Critically, this is INVISIBLE to the sample-counter integrity check reported in
DEVIATIONS.md (2026-06-11): no samples are dropped. The radio delivers every
sample. What degrades is WHEN the packets arrive, which the counter cannot see.

WHAT THIS MODULE PROVIDES
---------------------------------------------------------------------------
    fit_clock()               OLS and upper-envelope clock fits
    recording_residuals()     per-recording residual structure
    scan_all_recordings()     the residual sweep across the whole dataset
    realigned_recording()     a Recording re-aligned under a chosen fit
    score_recording()         AUC + balanced accuracy under a chosen fit,
                              scored with that subject's saved v3 model
    lag_sweep()               constant-offset scan, to show that a fixed shift
                              does NOT explain the failure

Everything is read-only with respect to the registered pipeline. Nothing here
modifies analysis/loader.py, and no derived data is overwritten. Re-aligned
epochs are built by handing corrected markers to the REGISTERED
preprocess_recording(), so filtering, epoching, baseline correction, boundary
rejection and the +/-150 uV gate are all exactly as pre-registered.

IMPORTANT -- ON SELECTIVE CORRECTION
---------------------------------------------------------------------------
Re-aligning only the recordings that produced inconvenient results would be
outcome-dependent correction and is indefensible. If the upper-envelope fit is
ever adopted for reported numbers it must be applied uniformly to all
recordings. This module is a diagnostic; it does not change any reported value.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from analysis.classifier import DOWNSAMPLED_HZ, _extract_features
from analysis.loader import SAMPLE_RATE_HZ, load_recording
from analysis.preprocess import preprocess_recording

warnings.filterwarnings("ignore")

CONDITIONS = ["control", "chewing", "emi", "acoustic"]

# Upper-envelope fit parameters. `keep` is the fraction of least-delayed points
# retained at each iteration; the fit converges onto the low-delay envelope.
ENVELOPE_KEEP = 0.30
ENVELOPE_ITERS = 8

# Residual std above which a recording is flagged as showing the pathology.
# Descriptive only: chosen as ~5x the dataset median (~6 ms), not a test.
FLAG_RESIDUAL_MS = 30.0

DERIVED_PIPELINE = "timing-diagnostic"


# ---------------------------------------------------------------------------
# Clock fitting
# ---------------------------------------------------------------------------

def fit_clock(
    ts_relative: np.ndarray,
    acq_time: np.ndarray,
    mode: str = "ols",
    keep: float = ENVELOPE_KEEP,
    iters: int = ENVELOPE_ITERS,
) -> tuple[float, float]:
    """Fit acq_time ~ m * ts_relative + c.

    mode="ols"            reproduces analysis/loader.py exactly.
    mode="upper_envelope" iteratively refits to the least-delayed points, i.e.
                          those with the highest residuals, since delay can only
                          push a point downward in acq-vs-arrival space.
    """
    m, c = np.polyfit(ts_relative, acq_time, 1)
    if mode == "ols":
        return float(m), float(c)
    if mode != "upper_envelope":
        raise ValueError(f"unknown mode: {mode}")

    for _ in range(iters):
        resid = acq_time - (m * ts_relative + c)
        threshold = np.quantile(resid, 1.0 - keep)
        sel = resid >= threshold
        if sel.sum() < 100:
            break
        m, c = np.polyfit(ts_relative[sel], acq_time[sel], 1)
    return float(m), float(c)


def _load_timing(subject_id: str, condition: str,
                 data_dir: str | Path = "data/raw") -> tuple[np.ndarray, np.ndarray]:
    """Return (ts_relative, acq_time) for one recording."""
    import json
    sub_dir = Path(data_dir) / f"sub-{subject_id}"
    stem = f"sub-{subject_id}_cond-{condition}"
    acq = np.load(sub_dir / f"{stem}_acqtime.npy")
    ts = np.load(sub_dir / f"{stem}_timestamps.npy")
    with open(sub_dir / f"{stem}_meta.json") as f:
        anchor = json.load(f)["wall_clock_anchor_unix"]
    return ts - anchor, acq


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def recording_residuals(subject_id: str, condition: str,
                        data_dir: str | Path = "data/raw") -> dict:
    """Residual structure of the OLS clock fit for one recording, in ms.

    A healthy recording has a small std and a mildly negative-skewed residual.
    The pathology shows as a large std with a long NEGATIVE tail -- the
    signature of one-sided delay.
    """
    tr, acq = _load_timing(subject_id, condition, data_dir)
    m, c = fit_clock(tr, acq, mode="ols")
    resid_ms = (acq - (m * tr + c)) * 1000.0
    p1, p25, p50, p75, p99 = np.percentile(resid_ms, [1, 25, 50, 75, 99])
    return {
        "subject": subject_id,
        "condition": condition,
        "residual_std_ms": float(resid_ms.std()),
        "residual_iqr_ms": float(p75 - p25),
        "p1_ms": float(p1), "p50_ms": float(p50), "p99_ms": float(p99),
        "min_ms": float(resid_ms.min()), "max_ms": float(resid_ms.max()),
        "drift_ms_per_s": float((1.0 - m) * 1000.0),
        "n_packets": int(len(acq)),
        "flagged": bool(resid_ms.std() > FLAG_RESIDUAL_MS),
    }


def analysed_subjects(derived_root: str | Path = "data/derived") -> set[str]:
    """Subjects retained for analysis, identified by a saved classifier model."""
    import glob as _glob
    pattern = str(Path(derived_root) / "classifier-v3" / "sub-*" / "sub-*_model.pkl")
    return {Path(p).parts[-2].replace("sub-", "") for p in _glob.glob(pattern)}


def scan_all_recordings(data_dir: str | Path = "data/raw",
                        analysed_only: bool = True,
                        derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """Residual diagnostics per recording.

    `data/raw` also holds partial recordings from participants screened out at the
    electrode-fit gate: at the time of writing, three such participants contribute
    four recordings. Including them gives 168 rows for a study of 41 participants,
    which produces per-condition counts (41/43/43/41) that do not reconcile with
    the reported N and invite a question the answer to which is uninformative.

    `analysed_only=True` therefore restricts the scan to the 41 retained
    participants, giving a clean 164 = 41 x 4. This is a presentational choice
    only: none of the excluded recordings is flagged (all sit at 5.6-5.9 ms), the
    flagged counts, the threshold counts, and the subject sets returned by
    analysis.robustness_checks.subjects_above_residual() are identical either way.
    Pass False to scan everything on disk.
    """
    keep = analysed_subjects(derived_root) if analysed_only else None
    rows = []
    for sub_dir in sorted(Path(data_dir).glob("sub-*")):
        sid = sub_dir.name.replace("sub-", "")
        if sid.startswith("pilot"):
            continue
        if keep is not None and sid not in keep:
            continue
        for cond in CONDITIONS:
            if not (sub_dir / f"sub-{sid}_cond-{cond}_acqtime.npy").exists():
                continue
            try:
                rows.append(recording_residuals(sid, cond, data_dir))
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Re-alignment and scoring
# ---------------------------------------------------------------------------

def realigned_recording(subject_id: str, condition: str, mode: str = "upper_envelope",
                        data_dir: str | Path = "data/raw"):
    """Load a Recording with markers re-aligned under the chosen clock fit.

    mode="ols" returns the recording exactly as analysis/loader.py produced it.
    """
    rec = load_recording(subject_id, condition, data_dir)
    if mode == "ols":
        return rec

    tr, acq = _load_timing(subject_id, condition, data_dir)
    m, c = fit_clock(tr, acq, mode=mode)

    markers = rec.markers.copy()
    anchor = rec.meta["recorder"]["wall_clock_anchor_unix"]
    marker_acq = m * (markers["wall_time"].values - anchor) + c
    markers["acq_time"] = marker_acq
    markers["sample"] = np.clip(
        np.round(marker_acq * SAMPLE_RATE_HZ).astype(int), 0, len(acq) - 1
    )
    return replace(rec, markers=markers, clock_fit=(m, c))


def _epochs_for(subject_id: str, condition: str, mode: str,
                data_dir: str | Path = "data/raw") -> mne.Epochs:
    """Re-aligned epochs, built by the REGISTERED preprocessing pipeline."""
    rec = realigned_recording(subject_id, condition, mode, data_dir)
    return preprocess_recording(rec, save=False).epochs


def _load_model(subject_id: str, derived_root: str | Path = "data/derived"):
    path = (Path(derived_root) / "classifier-v3" / f"sub-{subject_id}"
            / f"sub-{subject_id}_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def score_recording(subject_id: str, condition: str, mode: str = "ols",
                    data_dir: str | Path = "data/raw",
                    derived_root: str | Path = "data/derived") -> dict:
    """Score one recording under a chosen clock fit with that subject's v3 model.

    AUC is threshold-free and therefore the cleanest readout of whether the
    discriminative information survived. Balanced accuracy is reported too since
    it is the manuscript's metric. Note that for the control condition the model
    was trained on (part of) these same epochs, so control scores are optimistic
    by construction and are useful here only as a specificity check -- i.e. that
    re-alignment does not DAMAGE a healthy recording.
    """
    epochs = _epochs_for(subject_id, condition, mode, data_dir)
    X, y = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)
    clf = _load_model(subject_id, derived_root)
    scores = clf.decision_function(X)
    return {
        "subject": subject_id, "condition": condition, "mode": mode,
        "auc": float(roc_auc_score(y, scores)) if y.sum() > 0 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, clf.predict(X))) * 100,
        "n_epochs": int(len(epochs)),
        "n_target": int(y.sum()),
    }


def compare_fits(pairs: list[tuple[str, str]],
                 data_dir: str | Path = "data/raw",
                 derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """OLS vs upper-envelope for a list of (subject, condition) pairs."""
    rows = []
    for sid, cond in pairs:
        a = score_recording(sid, cond, "ols", data_dir, derived_root)
        b = score_recording(sid, cond, "upper_envelope", data_dir, derived_root)
        rows.append({
            "subject": sid, "condition": cond,
            "auc_ols": round(a["auc"], 3),
            "auc_envelope": round(b["auc"], 3),
            "auc_delta": round(b["auc"] - a["auc"], 3),
            "balacc_ols": round(a["balanced_accuracy"], 1),
            "balacc_envelope": round(b["balanced_accuracy"], 1),
            "n_epochs_ols": a["n_epochs"], "n_epochs_env": b["n_epochs"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constant-lag control analysis
# ---------------------------------------------------------------------------

def _fit_clock_any(tr: np.ndarray, acq: np.ndarray, estimator: str) -> tuple[float, float]:
    """Clock fit under a named estimator. Extends fit_clock() to robust regressors."""
    if estimator in ("ols", "upper_envelope"):
        return fit_clock(tr, acq, mode=estimator)

    from sklearn.linear_model import (HuberRegressor, LinearRegression,
                                      RANSACRegressor)
    X = tr.reshape(-1, 1)
    if estimator == "huber":
        m = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=300).fit(X, acq)
        return float(m.coef_[0]), float(m.intercept_)
    if estimator == "ransac":
        m = RANSACRegressor(LinearRegression(), random_state=0, max_trials=100).fit(X, acq)
        return float(m.estimator_.coef_[0]), float(m.estimator_.intercept_)
    raise ValueError(f"unknown estimator: {estimator}")


def _score_with_fit(subject_id: str, condition: str, m: float, c: float,
                    data_dir: str | Path, derived_root: str | Path) -> float:
    """AUC for a recording whose markers are placed by an arbitrary clock fit."""
    from sklearn.metrics import roc_auc_score
    rec = load_recording(subject_id, condition, data_dir)
    anchor = rec.meta["recorder"]["wall_clock_anchor_unix"]
    markers = rec.markers.copy()
    marker_acq = m * (markers["wall_time"].values - anchor) + c
    markers["acq_time"] = marker_acq
    markers["sample"] = np.clip(
        np.round(marker_acq * SAMPLE_RATE_HZ).astype(int), 0, len(rec.acq_time) - 1)
    epochs = preprocess_recording(replace(rec, markers=markers), save=False).epochs
    X, y = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)
    clf = _load_model(subject_id, derived_root)
    return float(roc_auc_score(y, clf.decision_function(X)))


ESTIMATORS = ("ols", "upper_envelope", "huber", "ransac")


def compare_estimators(pairs: list[tuple[str, str]],
                       estimators: tuple[str, ...] = ESTIMATORS,
                       data_dir: str | Path = "data/raw",
                       derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """AUC per recording under each clock estimator.

    The upper-envelope fit targets the low-delay boundary directly and is the
    most aggressive of the four; Huber downweights outliers while remaining a
    central-tendency estimator, so it degrades gracefully to OLS when residuals
    are symmetric. That difference is the whole point of running this: an
    estimator that repairs badly misaligned recordings while leaving well-aligned
    ones untouched is preferable to one that trades the second for the first.
    """
    rows = []
    for sid, cond in pairs:
        tr, acq = _load_timing(sid, cond, data_dir)
        row = {"subject": sid, "condition": cond,
               "residual_std_ms": recording_residuals(sid, cond, data_dir)["residual_std_ms"]}
        for est in estimators:
            try:
                m, c = _fit_clock_any(tr, acq, est)
                row[f"auc_{est}"] = round(_score_with_fit(sid, cond, m, c,
                                                          data_dir, derived_root), 4)
            except Exception:
                row[f"auc_{est}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def keep_sensitivity(pairs: list[tuple[str, str]],
                     keeps: tuple[float, ...] = (0.15, 0.20, 0.30, 0.40, 0.50),
                     data_dir: str | Path = "data/raw",
                     derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """AUC across values of the upper-envelope `keep` fraction.

    ENVELOPE_KEEP = 0.30 was chosen without tuning. This exists so that the
    dependence of any conclusion on that unmotivated choice is visible rather
    than assumed away.
    """
    rows = []
    for sid, cond in pairs:
        tr, acq = _load_timing(sid, cond, data_dir)
        row = {"subject": sid, "condition": cond,
               "residual_std_ms": recording_residuals(sid, cond, data_dir)["residual_std_ms"]}
        for k in keeps:
            m, c = fit_clock(tr, acq, mode="upper_envelope", keep=k)
            row[f"keep_{k:.2f}"] = round(
                _score_with_fit(sid, cond, m, c, data_dir, derived_root), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def residual_shape(pairs: list[tuple[str, str]] | None = None,
                   data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Skewness and excess kurtosis of each recording's OLS residuals.

    Used to test whether the direction of change under the envelope fit tracks
    how one-sided a recording's delay distribution is. It does not; see
    notebooks/timing_diagnostic.ipynb section 9.
    """
    from scipy import stats as _st
    if pairs is None:
        scan = scan_all_recordings(data_dir)
        pairs = list(zip(scan["subject"], scan["condition"]))
    rows = []
    for sid, cond in pairs:
        tr, acq = _load_timing(sid, cond, data_dir)
        m, c = fit_clock(tr, acq, "ols")
        r = acq - (m * tr + c)
        rows.append({"subject": sid, "condition": cond,
                     "residual_skew": float(_st.skew(r)),
                     "residual_kurtosis": float(_st.kurtosis(r))})
    return pd.DataFrame(rows)


def sweep_recording(subject_id: str, condition: str,
                    data_dir: str | Path = "data/raw",
                    derived_root: str | Path = "data/derived") -> dict:
    """Score one recording under both clock fits and return a flat result dict.

    This is the unit of work for the full-dataset sweep. Scoring uses that
    subject's saved v3 model, which was trained on their control epochs under the
    ORIGINAL alignment.

    CONTROL RECORDINGS ARE CONFOUNDED BY THIS and are marked accordingly.
    For a noise recording the model is fixed and only the test data moves, which
    is a clean comparison. For a control recording, re-alignment moves the very
    epochs the model was trained on, so a drop may reflect train/test mismatch
    rather than lost signal. Any recovery claim should be made on non-control
    recordings, or the model retrained on re-aligned control epochs first.
    """
    import time
    t0 = time.time()
    resid = recording_residuals(subject_id, condition, data_dir)
    a = score_recording(subject_id, condition, "ols", data_dir, derived_root)
    b = score_recording(subject_id, condition, "upper_envelope", data_dir, derived_root)
    return {
        "subject": subject_id,
        "condition": condition,
        "is_control": condition == "control",
        "confounded": condition == "control",
        "residual_std_ms": resid["residual_std_ms"],
        "residual_p1_ms": resid["p1_ms"],
        "flagged": resid["flagged"],
        "auc_ols": a["auc"], "auc_envelope": b["auc"],
        "auc_delta": b["auc"] - a["auc"],
        "balacc_ols": a["balanced_accuracy"],
        "balacc_envelope": b["balanced_accuracy"],
        "balacc_delta": b["balanced_accuracy"] - a["balanced_accuracy"],
        "n_epochs_ols": a["n_epochs"], "n_epochs_envelope": b["n_epochs"],
        "elapsed_s": round(time.time() - t0, 2),
    }


def load_sweep(derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """Load every per-recording JSON written by analysis/run_timing_sweep.py."""
    import json
    d = Path(derived_root) / DERIVED_PIPELINE / "per_recording"
    rows = []
    for p in sorted(d.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_cause_frame(data_dir: str | Path = "data/raw",
                      orders_csv: str | Path = "protocol/order_assignments/order_assignments.csv",
                      ) -> pd.DataFrame:
    """One row per recording, joining residual diagnostics to candidate causes.

    Columns added alongside the residual measures:
        late_frame_pct  pygame dropped-frame rate -- a HOST-side stall indicator.
                        The stimulus loop and the UDP receiver share a machine, so
                        a scheduling stall severe enough to delay packets by
                        hundreds of ms should also drop frames at 60 Hz.
        battery_mean    headset battery over the recording (transmit-power proxy).
        condition_order position of the recording within its session (1-4).
        date            session date, for temporal clustering.
        samples_dropped counter-derived loss; zero throughout the dataset.
    """
    import json

    scan = scan_all_recordings(data_dir)
    rows = []
    for sub_dir in sorted(Path(data_dir).glob("sub-*")):
        sid = sub_dir.name.replace("sub-", "")
        if sid.startswith("pilot"):
            continue
        for cond in CONDITIONS:
            stem = f"sub-{sid}_cond-{cond}"
            if not (sub_dir / f"{stem}_eeg.npy").exists():
                continue
            try:
                with open(sub_dir / f"{stem}_session.json") as f:
                    sess = json.load(f)
                with open(sub_dir / f"{stem}_meta.json") as f:
                    meta = json.load(f)
            except Exception:
                continue
            late, total = sess.get("late_frames"), sess.get("total_frames")
            battery = np.load(sub_dir / f"{stem}_eeg.npy", mmap_mode="r")[:, 14]
            rows.append({
                "subject": sid, "condition": cond,
                "late_frame_pct": (100.0 * late / total) if total else np.nan,
                "battery_mean": float(np.asarray(battery, dtype=float).mean()),
                "samples_dropped": meta.get("samples_dropped", np.nan),
                "date": str(meta.get("start_time_iso_utc", ""))[:10],
            })

    df = scan.merge(pd.DataFrame(rows), on=["subject", "condition"], how="inner")
    try:
        orders = pd.read_csv(orders_csv)
        orders["subject"] = orders["subject_id"].str.replace("sub-", "", regex=False)
        df = df.merge(orders[["subject", "condition", "condition_order"]],
                      on=["subject", "condition"], how="left")
    except Exception:
        df["condition_order"] = np.nan
    return df


def cause_analysis(df: pd.DataFrame | None = None, **kwargs) -> dict:
    """Test candidate causes of the alignment fault against the recording set.

    Each candidate predicts a specific association; none of them holds:

        host scheduling   -> flagged recordings should drop stimulus frames
        transmit power    -> flagged recordings should show lower battery
        progressive wear  -> flagged recordings should fall late in a session
        the manipulation  -> flagged recordings should concentrate in one condition

    What does hold is session-level clustering: affected recordings arrive in
    contiguous runs within a session and then stop, which is the signature of an
    episodic degradation of the wireless path rather than a property of the
    headset, the host, or the condition.
    """
    from scipy.stats import mannwhitneyu, spearmanr

    if df is None:
        df = build_cause_frame(**kwargs)
    flagged, normal = df[df.flagged], df[~df.flagged]

    def _assoc(col: str, alternative: str) -> dict:
        a = flagged[col].dropna()
        b = normal[col].dropna()
        rho, p_rho = spearmanr(df[col].fillna(df[col].median()), df["residual_std_ms"])
        try:
            _, p_mw = mannwhitneyu(a, b, alternative=alternative)
        except Exception:
            p_mw = float("nan")
        return {"flagged_mean": float(a.mean()), "normal_mean": float(b.mean()),
                "spearman_rho_vs_residual": round(float(rho), 3),
                "spearman_p": float(p_rho), "mannwhitney_p": float(p_mw)}

    per_subject = df.groupby("subject")["flagged"].sum()
    n_multi = int((per_subject >= 2).sum())
    n_flagged = int(df["flagged"].sum())
    # Expected number of subjects with >=2 hits if flagged recordings were
    # scattered independently across recordings.
    n_sub, k = df["subject"].nunique(), len(df)
    p_hit = n_flagged / k
    expected_multi = n_sub * (1 - (1 - p_hit) ** 4 - 4 * p_hit * (1 - p_hit) ** 3)

    return {
        "n_recordings": int(len(df)),
        "n_flagged": n_flagged,
        "host_scheduling": _assoc("late_frame_pct", "greater"),
        "transmit_power": _assoc("battery_mean", "less"),
        "progressive_wear": {
            "flagged_by_condition_order":
                df[df.flagged]["condition_order"].value_counts().sort_index().to_dict(),
            "all_by_condition_order":
                df["condition_order"].value_counts().sort_index().to_dict(),
            "spearman_rho_vs_residual": round(float(
                spearmanr(df["condition_order"].fillna(0), df["residual_std_ms"])[0]), 3),
        },
        "session_clustering": {
            "subjects_with_any": {k_: int(v) for k_, v in per_subject[per_subject > 0].items()},
            "n_subjects_with_two_or_more": n_multi,
            "expected_if_independent": round(float(expected_multi), 2),
        },
        "samples_dropped_total": float(df["samples_dropped"].sum()),
    }


def lag_sweep(subject_id: str, condition: str, lags_ms=None,
              data_dir: str | Path = "data/raw",
              derived_root: str | Path = "data/derived") -> pd.DataFrame:
    """AUC as a function of a CONSTANT marker shift applied to the OLS alignment.

    Two things fall out of this:

    1. On a healthy recording, AUC peaks sharply at zero lag and falls BELOW
       chance at +/- one SOA (233 ms). Below-chance AUC is diagnostic: additive
       noise cannot produce it, but a one-flash marker offset can, because the
       epoch then contains the neighbouring flash's response.

    2. On the two collapsed recordings, no constant lag restores performance.
       The misalignment is therefore time-varying, not a fixed offset -- which is
       what the one-sided-delay mechanism predicts and what motivates refitting
       the clock rather than shifting the markers.
    """
    if lags_ms is None:
        lags_ms = [-466, -233, -117, 0, 117, 233, 466]

    rec = load_recording(subject_id, condition, data_dir)
    clf = _load_model(subject_id, derived_root)
    base = rec.markers.reset_index(drop=True)

    rows = []
    for lag in lags_ms:
        shifted = base.copy()
        shifted["sample"] = np.clip(
            shifted["sample"].astype(int).values + int(round(lag / 1000 * SAMPLE_RATE_HZ)),
            0, len(rec.acq_time) - 1,
        )
        epochs = preprocess_recording(replace(rec, markers=shifted), save=False).epochs
        if len(epochs) < 50:
            rows.append({"lag_ms": lag, "auc": np.nan, "n_epochs": len(epochs)})
            continue
        X, y = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)
        auc = roc_auc_score(y, clf.decision_function(X)) if y.sum() > 4 else np.nan
        rows.append({"lag_ms": lag, "auc": float(auc), "n_epochs": int(len(epochs))})
    return pd.DataFrame(rows)
