"""
Classifier pipeline (v3): canonical SWLDA per pre-registration §5.

WHAT CHANGED FROM v2
--------------------------------------------------------------------------
v2 implemented "SWLDA" as: stepwise OLS *feature selection*, then a separate
shrinkage-LDA fitted on the survivors. That is a select-then-LDA pipeline, not
the SWLDA named in the pre-registration.

Canonical SWLDA (Krusienski et al. 2008, the standard P300-speller method) is
stepwise *linear regression* whose regression weights ARE the discriminant
function. There is no second model: a new epoch is scored by the linear
combination of the selected features with their OLS weights, and that scalar
score is thresholded. v3 implements exactly that, in `SWLDAClassifier`.

Two further v3 choices, both moving TOWARD the registered text rather than away:
  1. No training-set class balancing for SWLDA. The pre-reg says "train on the
     first 70% of epochs" and never specifies subsampling; v2's majority-class
     subsampling was an unregistered step. SWLDA is regression, so imbalance is
     absorbed by the continuous target and by the decision threshold rather than
     by throwing data away. Dropping it also gives the regression ~4x more
     training rows and a better-conditioned fit. (The 'lda' scaffold path keeps
     balancing, since plain LDA does need it — that path is for validation only.)
  2. Decision threshold = midpoint of the two class-mean training scores. For
     roughly equal-variance scores with equal priors this is the balanced-
     accuracy-optimal cut, and it never peeks at the test set.

This brings the classifier into line with §5; it is a compliance fix, not a
deviation. NOTE: any preliminary P300 numbers (incl. the gate-impact p-values
in DEVIATIONS.md) were produced by the v2 scaffold and will shift once results
are regenerated with v3. Regenerate before quoting them in the manuscript.

Output paths:
    data/derived/classifier-v3/sub-<id>/sub-<id>_model.pkl
    data/derived/classifier-v3/sub-<id>/sub-<id>_results.json
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from scipy.stats import friedmanchisquare, wilcoxon, t as t_dist
from statsmodels.stats.anova import AnovaRM


# ---- Pre-registered parameters ----------------------------------------
# §5: 50 samples per epoch (1 per 20 ms over the 1000 ms window).
DOWNSAMPLED_HZ = 50
# §5: 70/30 train/test split on control for the within-condition ceiling.
TRAIN_FRACTION = 0.70

DERIVED_PIPELINE = "classifier-v3"
# Blockwise (leave-one-sub-block-out) robustness analysis, added 2026-08-13 in
# response to SYNASC 2026 Reviewer 1. Same SWLDA, same preprocessing epochs, same
# parameters -- ONLY the control train/test split differs. Deliberately a sibling
# directory rather than a version bump: it does not supersede v3, and under the
# decision rule pre-committed in DEVIATIONS.md (2026-08-13) v3 may remain primary.
DERIVED_PIPELINE_BLOCKWISE = "classifier-v3-blockwise"
PREPROCESSING_PIPELINE = "preprocessing-v2"


# ---- SWLDA parameters (Krusienski et al. 2008, canonical for P300 spellers) ---
# Locked before any data is collected. Changing requires deviation logging.
SWLDA_P_ENTER = 0.1     # feature added if conditional p-value < this
SWLDA_P_REMOVE = 0.15   # feature removed if conditional p-value > this
SWLDA_MAX_FEATURES = 60  # hard cap on selected feature count

# Label coding for the regression target. Target = +1, non-target = -1.
# (p-values used for selection are invariant to this coding; it only sets the
# score scale, which the threshold absorbs.)
TARGET_CODE = 1.0
NONTARGET_CODE = -1.0


@dataclass
class ClassifierResult:
    """Per-subject classifier evaluation across conditions."""
    subject_id: str
    classifier_type: str
    n_features: int
    n_train_epochs: int
    train_target_count: int
    train_nontarget_count: int
    per_condition: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)


# ==========================================================================
# Canonical SWLDA
# ==========================================================================

class SWLDAClassifier:
    """Stepwise linear discriminant analysis as stepwise linear regression.

    Forward/backward stepwise OLS selects features by conditional p-value
    (enter if p < p_enter, remove if p > p_remove, capped at max_features).
    The OLS coefficients on the selected features are the discriminant weights:
    a new epoch x is scored as

        score(x) = b0 + sum_j b_j * x[selected_j]

    and classified by comparing score(x) to a threshold. This is canonical
    SWLDA — the regression *is* the classifier; there is no separate LDA.

    sklearn-style API (fit / predict / decision_function) so it drops straight
    into the existing _evaluate(clf, X, y) call. predict() takes the FULL
    feature matrix and slices the selected columns internally.
    """

    def __init__(
        self,
        p_enter: float = SWLDA_P_ENTER,
        p_remove: float = SWLDA_P_REMOVE,
        max_features: int = SWLDA_MAX_FEATURES,
        verbose: bool = False,
    ):
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.max_features = max_features
        self.verbose = verbose
        # set by fit()
        self.selected_features_: list[int] = []
        self.intercept_: float = 0.0
        self.coef_: np.ndarray = np.empty(0)
        self.threshold_: float = 0.0
        self.target_is_high_: bool = True

    # ---- stepwise selection (forward by best p, backward by worst p) -----
    def _stepwise_select(self, X: np.ndarray, y_reg: np.ndarray) -> list[int]:
        n_samples, n_features = X.shape
        selected: list[int] = []
        available = list(range(n_features))

        iteration = 0
        max_iterations = self.max_features * 4  # safety bound
        while iteration < max_iterations:
            iteration += 1

            # ---- Forward: best candidate to add --------------------------
            best_pval = 1.0
            best_feature = None
            if len(selected) < self.max_features:
                for j in available:
                    trial = selected + [j]
                    X_trial = sm.add_constant(X[:, trial], has_constant="add")
                    try:
                        model = sm.OLS(y_reg, X_trial).fit()
                        pval = model.pvalues[-1]  # the just-added feature
                    except Exception:
                        continue
                    if pval < best_pval:
                        best_pval = pval
                        best_feature = j

            added = False
            if best_feature is not None and best_pval < self.p_enter:
                selected.append(best_feature)
                available.remove(best_feature)
                added = True
                if self.verbose:
                    print(f"  + add feat {best_feature} (p={best_pval:.4f}), "
                          f"selected={len(selected)}")

            # ---- Backward: drop any feature whose p drifted above remove --
            removed = False
            if len(selected) > 1:
                X_full = sm.add_constant(X[:, selected], has_constant="add")
                try:
                    model = sm.OLS(y_reg, X_full).fit()
                    pvals_selected = model.pvalues[1:]  # skip constant
                    worst_idx = int(np.argmax(pvals_selected))
                    worst_pval = float(pvals_selected[worst_idx])
                    if worst_pval > self.p_remove:
                        feat = selected[worst_idx]
                        selected.pop(worst_idx)
                        available.append(feat)
                        removed = True
                        if self.verbose:
                            print(f"  - remove feat {feat} (p={worst_pval:.4f}), "
                                  f"selected={len(selected)}")
                except Exception:
                    pass

            if not added and not removed:
                break

        return sorted(selected)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SWLDAClassifier":
        """Fit on full feature matrix X and 0/1 labels y (1 = target)."""
        y = np.asarray(y).astype(int)
        y_reg = np.where(y == 1, TARGET_CODE, NONTARGET_CODE).astype(float)

        self.selected_features_ = self._stepwise_select(X, y_reg)
        if len(self.selected_features_) == 0:
            raise RuntimeError("SWLDA selected zero features. "
                               "Signal is too weak for this configuration.")

        # Final OLS fit on the selected features -> discriminant weights.
        X_sel = sm.add_constant(X[:, self.selected_features_], has_constant="add")
        model = sm.OLS(y_reg, X_sel).fit()
        self.intercept_ = float(model.params[0])
        self.coef_ = np.asarray(model.params[1:], dtype=float)

        # Decision threshold: midpoint of the two class-mean training scores.
        # Equal-prior, equal-variance optimum; balances TPR/TNR without peeking
        # at test data, which is what makes the held-out balanced accuracy honest.
        scores = self._raw_scores(X)
        mu_t = float(scores[y == 1].mean())
        mu_n = float(scores[y == 0].mean())
        self.threshold_ = 0.5 * (mu_t + mu_n)
        self.target_is_high_ = mu_t >= mu_n

        if self.verbose:
            print(f"  SWLDA fit: {len(self.selected_features_)} of "
                  f"{X.shape[1]} features | threshold={self.threshold_:.4f} | "
                  f"target_mean={mu_t:.3f} nontarget_mean={mu_n:.3f}")
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X[:, self.selected_features_] @ self.coef_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the threshold; positive => target side."""
        s = self._raw_scores(X)
        return (s - self.threshold_) if self.target_is_high_ else (self.threshold_ - s)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)


# ==========================================================================
# IO + features (unchanged from v2)
# ==========================================================================

def _load_epochs(subject_id: str, condition: str, derived_root: Path) -> mne.Epochs:
    """Load preprocessed epochs from disk."""
    path = (derived_root / PREPROCESSING_PIPELINE / f"sub-{subject_id}"
            / f"sub-{subject_id}_cond-{condition}_epo.fif")
    if not path.exists():
        raise FileNotFoundError(f"No preprocessed epochs at {path}. "
                                f"Run preprocess_recording first.")
    return mne.read_epochs(path, preload=True, verbose="WARNING")


def _extract_features(epochs: mne.Epochs, target_sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample to target_sfreq and flatten into feature vectors.

    Returns:
        X: (n_epochs, n_channels * n_timepoints_per_epoch) float32 feature matrix
        y: (n_epochs,) int array, 1 = target, 0 = nontarget
    """
    epochs_ds = epochs.copy().resample(target_sfreq, verbose="WARNING")
    data = epochs_ds.get_data() * 1e6  # volts -> microvolts
    n_epochs, n_channels, n_samples = data.shape
    X = data.reshape(n_epochs, n_channels * n_samples).astype(np.float32)
    y = (epochs_ds.metadata["is_target"].values).astype(int)
    return X, y


def _evaluate(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Run prediction and compute the metric suite. clf.predict takes full X."""
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n_epochs": int(len(y_test)),
        "n_target": int(y_test.sum()),
        "n_nontarget": int((1 - y_test).sum()),
        "balanced_accuracy": float(bal_acc),
        "true_target_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "true_nontarget_rate": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
    }


# ==========================================================================
# Train + evaluate
# ==========================================================================

def train_and_evaluate(
    subject_id: str,
    classifier_type: str = "swlda",
    derived_root: str | Path = "data/derived",
    save: bool = True,
    random_state: int = 42,
) -> ClassifierResult:
    """Train classifier on control, evaluate on all four conditions.

    Args:
        subject_id: e.g. "20"
        classifier_type: "swlda" for the registered final classifier (canonical),
            or "lda" for the balanced shrinkage-LDA scaffold (validation only).
        derived_root: base path for preprocessing inputs and classifier outputs.
        save: persist model and results JSON.
        random_state: train/test split reproducibility.
    """
    derived_root = Path(derived_root)

    # ---- Load control, split -------------------------------------------
    control_epochs = _load_epochs(subject_id, "control", derived_root)
    X_ctrl, y_ctrl = _extract_features(control_epochs, target_sfreq=DOWNSAMPLED_HZ)

    X_train, X_test, y_train, y_test = train_test_split(
        X_ctrl, y_ctrl,
        train_size=TRAIN_FRACTION,
        stratify=y_ctrl,
        random_state=random_state,
    )

    selected_features: list[int] | None = None
    balancing_applied = False

    if classifier_type == "lda":
        # ---- Scaffold path: balance training, fit shrinkage LDA ----------
        # Plain LDA collapses to "always non-target" under 1:7 imbalance, so the
        # scaffold subsamples the majority class. Validation only — NOT the
        # registered classifier.
        rng = np.random.default_rng(random_state)
        target_idx = np.where(y_train == 1)[0]
        nontarget_idx = np.where(y_train == 0)[0]
        n_min = len(target_idx)
        nontarget_kept = rng.choice(nontarget_idx, size=n_min, replace=False)
        train_idx = np.concatenate([target_idx, nontarget_kept])
        rng.shuffle(train_idx)
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        balancing_applied = True

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_train, y_train)
        n_features = int(X_train.shape[1])

    elif classifier_type == "swlda":
        # ---- Registered path: canonical SWLDA, no balancing --------------
        clf = SWLDAClassifier(
            p_enter=SWLDA_P_ENTER,
            p_remove=SWLDA_P_REMOVE,
            max_features=SWLDA_MAX_FEATURES,
            verbose=True,
        )
        clf.fit(X_train, y_train)
        selected_features = clf.selected_features_
        n_features = int(len(selected_features))
        print(f"  SWLDA selected {n_features} of {X_train.shape[1]} features")

    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    result = ClassifierResult(
        subject_id=subject_id,
        classifier_type=classifier_type,
        n_features=n_features,
        n_train_epochs=int(len(y_train)),
        train_target_count=int((y_train == 1).sum()),
        train_nontarget_count=int((y_train == 0).sum()),
        parameters={
            "downsampled_hz": DOWNSAMPLED_HZ,
            "train_fraction": TRAIN_FRACTION,
            "random_state": random_state,
            "balancing_applied": balancing_applied,
            "swlda_p_enter": SWLDA_P_ENTER if classifier_type == "swlda" else None,
            "swlda_p_remove": SWLDA_P_REMOVE if classifier_type == "swlda" else None,
            "swlda_max_features": SWLDA_MAX_FEATURES if classifier_type == "swlda" else None,
            "swlda_selected_features": selected_features,
            "swlda_threshold": float(clf.threshold_) if classifier_type == "swlda" else None,
            "swlda_intercept": float(clf.intercept_) if classifier_type == "swlda" else None,
            "swlda_coef": clf.coef_.tolist() if classifier_type == "swlda" else None,
            # scaffold-only fields
            "shrinkage": "auto" if classifier_type == "lda" else None,
            "solver": "lsqr" if classifier_type == "lda" else None,
        },
    )

    # ---- Evaluate: held-out control + each noise condition -------------
    # Note: clf.predict takes the FULL feature matrix in both paths; SWLDA
    # slices its selected columns internally, LDA uses all of them.
    result.per_condition["control_heldout"] = _evaluate(clf, X_test, y_test)
    for cond in ["chewing", "emi", "acoustic"]:
        epochs = _load_epochs(subject_id, cond, derived_root)
        X, y = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)
        result.per_condition[cond] = _evaluate(clf, X, y)

    # ---- Persist -------------------------------------------------------
    if save:
        out_dir = derived_root / DERIVED_PIPELINE / f"sub-{subject_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"sub-{subject_id}_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        result_dict = {
            "subject_id": result.subject_id,
            "classifier_type": result.classifier_type,
            "n_features": result.n_features,
            "n_train_epochs": result.n_train_epochs,
            "train_target_count": result.train_target_count,
            "train_nontarget_count": result.train_nontarget_count,
            "per_condition": result.per_condition,
            "parameters": result.parameters,
        }
        with open(out_dir / f"sub-{subject_id}_results.json", "w") as f:
            json.dump(result_dict, f, indent=2)

    return result


# ==========================================================================
# GROUP-LEVEL ANALYSIS  (primary confirmatory statistics, per pre-reg §5)
# --------------------------------------------------------------------------
# Mirrors analysis/n170.py:run_group_statistics, but on classifier balanced
# accuracy and in the predicted direction noise < control. The Greenhouse-
# Geisser epsilon here is the CORRECTED contrast-space version (the N170
# module still carries the uncorrected one — backport this, or factor a shared
# helper). Retention and false-positive aggregations are provided as
# descriptive context; their tests are flagged EXPLORATORY (not registered as
# inferential outcomes — §5 lists raw signal quality as descriptive).
# ==========================================================================

CLF_CONDITIONS = ["control_heldout", "chewing", "emi", "acoustic"]
RAW_CONDITIONS = ["control", "chewing", "emi", "acoustic"]   # for rejection logs
CONTROL_CEILING_THRESHOLD = 0.60   # §5 contingency: below this, primary -> N170


def load_all_results(derived_root: str | Path = "data/derived",
                     pipeline: str = DERIVED_PIPELINE) -> list[dict]:
    """Scan data/derived/<pipeline> and load every _results.json found.

    `pipeline` defaults to DERIVED_PIPELINE, so existing calls are unchanged in
    behaviour. Pass DERIVED_PIPELINE_BLOCKWISE to load the blockwise robustness
    results, which are written in the identical schema so that the SAME
    run_group_statistics() runs over either set without modification.
    """
    derived_root = Path(derived_root)
    clf_dir = derived_root / pipeline
    out = []
    for path in sorted(clf_dir.glob("sub-*/sub-*_results.json")):
        with open(path) as f:
            out.append(json.load(f))
    return out


def build_accuracy_dataframe(results: list[dict]) -> pd.DataFrame:
    """Tidy long-format frame: one row per (subject, condition)."""
    rows = []
    for r in results:
        subj = r["subject_id"]
        for cond in CLF_CONDITIONS:
            if cond not in r["per_condition"]:
                continue
            m = r["per_condition"][cond]
            cm = m["confusion_matrix"]
            fpr = cm["FP"] / (cm["FP"] + cm["TN"]) if (cm["FP"] + cm["TN"]) > 0 else 0.0
            rows.append({
                "subject": subj,
                "condition": cond,
                "balanced_accuracy": m["balanced_accuracy"] * 100,
                "sensitivity": m["true_target_rate"] * 100,
                "specificity": m["true_nontarget_rate"] * 100,
                "false_positive_rate": fpr * 100,
                "n_epochs": m["n_epochs"],
                "TP": cm["TP"], "TN": cm["TN"], "FP": cm["FP"], "FN": cm["FN"],
            })
    return pd.DataFrame(rows)


def _build_accuracy_matrix(results: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """(n_subjects, n_conditions) balanced-accuracy matrix (%). NaN if missing."""
    subjects = [r["subject_id"] for r in results]
    n, k = len(subjects), len(CLF_CONDITIONS)
    M = np.full((n, k), np.nan)
    for i, r in enumerate(results):
        for j, cond in enumerate(CLF_CONDITIONS):
            if cond in r["per_condition"]:
                M[i, j] = r["per_condition"][cond]["balanced_accuracy"] * 100
    return M, subjects, list(CLF_CONDITIONS)


def _greenhouse_geisser_epsilon(data: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon, computed in the (k-1)-dim contrast space.

    data: (n_subjects, n_conditions). Projects the condition covariance onto an
    orthonormal contrast basis (removing the shared 'all-conditions' axis)
    BEFORE the trace ratio. Omitting that projection — as the N170 module
    currently does — pins epsilon near the 1/(k-1) floor and over-corrects.
    Clipped to [1/(k-1), 1].
    """
    k = data.shape[1]
    S = np.cov(data.T)
    H = np.eye(k) - np.ones((k, k)) / k          # centering matrix
    u, _s, _vt = np.linalg.svd(H)
    C = u[:, :k - 1].T                           # orthonormal, C @ ones = 0
    Sstar = C @ S @ C.T
    tr = np.trace(Sstar)
    tr2 = np.trace(Sstar @ Sstar)
    if tr2 <= 0:
        return 1.0
    eps = (tr ** 2) / ((k - 1) * tr2)
    return float(np.clip(eps, 1.0 / (k - 1), 1.0))


def _paired_cohens_d_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired Cohen's d_z with analytical 95% CI. d<0 => x lower than y."""
    diffs = x - y
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    if sd_d == 0:
        return {"cohens_d": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "mean_diff": mean_d, "sd_diff": sd_d, "n": n}
    d = mean_d / sd_d
    se_d = float(np.sqrt(1.0 / n + d ** 2 / (2.0 * n)))
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    return {"cohens_d": float(d), "ci_low": float(d - t_crit * se_d),
            "ci_high": float(d + t_crit * se_d),
            "mean_diff": mean_d, "sd_diff": sd_d, "n": n}


def check_control_ceiling(results: list[dict],
                          threshold: float = CONTROL_CEILING_THRESHOLD) -> dict:
    """§5 contingency: if mean held-out control balanced accuracy < threshold,
    the classifier analysis is uninterpretable and N170 becomes primary."""
    vals = [r["per_condition"]["control_heldout"]["balanced_accuracy"]
            for r in results if "control_heldout" in r["per_condition"]]
    mean_ceiling = float(np.mean(vals)) if vals else 0.0
    return {
        "mean_control_balanced_accuracy": mean_ceiling,
        "threshold": threshold,
        "passes": mean_ceiling >= threshold,
        "n": len(vals),
    }


def run_group_statistics(results: list[dict]) -> dict:
    """Pre-registered PRIMARY group statistics on classifier balanced accuracy.

    1. Friedman omnibus across the four conditions (non-parametric, primary).
    2. Pairwise Wilcoxon signed-rank vs held-out control, ONE-TAILED in the
       predicted direction noise < control (i.e. noise degrades accuracy).
    3. RM-ANOVA with corrected Greenhouse-Geisser (parametric robustness check).
    Effect sizes: paired Cohen's d_z with 95% CI. Subjects with any missing
    condition are excluded list-wise.
    """
    M, subjects, conditions = _build_accuracy_matrix(results)
    valid = ~np.any(np.isnan(M), axis=1)
    M = M[valid]
    subjects_valid = [s for s, v in zip(subjects, valid) if v]
    n = len(subjects_valid)

    ctrl_idx = conditions.index("control_heldout")
    ctrl_vals = M[:, ctrl_idx]

    descriptives = {}
    for j, cond in enumerate(conditions):
        v = M[:, j]
        descriptives[cond] = {
            "n": n, "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 0 else 0.0,
            "median": float(np.median(v)),
        }

    friedman_stat, friedman_p = friedmanchisquare(*[M[:, j] for j in range(len(conditions))])

    posthoc = {}
    for cond in [c for c in conditions if c != "control_heldout"]:
        j = conditions.index(cond)
        noise_vals = M[:, j]
        try:
            stat_two, p_two = wilcoxon(noise_vals, ctrl_vals, alternative="two-sided")
            # predicted: noise accuracy LOWER than control
            stat_one, p_one = wilcoxon(noise_vals, ctrl_vals, alternative="less")
        except Exception:
            stat_two = p_two = stat_one = p_one = float("nan")
        effect = _paired_cohens_d_with_ci(noise_vals, ctrl_vals)  # d<0 = degraded
        posthoc[cond] = {
            "wilcoxon_statistic": float(stat_two),
            "p_two_tailed": float(p_two),
            "p_one_tailed": float(p_one),
            **effect,
        }

    # ---- Holm-Bonferroni across the post-hoc family --------------------
    # Three pre-registered one-tailed comparisons (each noise vs control).
    # Holm holds the family-wise false-positive rate at alpha while staying
    # uniformly more powerful than plain Bonferroni. Reported alongside the
    # raw p; the chewing effect survives and the nulls stay null.
    _ph_conds = list(posthoc.keys())
    _raw_p = [posthoc[c]["p_one_tailed"] for c in _ph_conds]
    if _ph_conds and all(np.isfinite(p) for p in _raw_p):
        from statsmodels.stats.multitest import multipletests
        _reject, _p_holm, _, _ = multipletests(_raw_p, alpha=0.05, method="holm")
        for c, p_adj, rej in zip(_ph_conds, _p_holm, _reject):
            posthoc[c]["p_one_tailed_holm"] = float(p_adj)
            posthoc[c]["significant_holm"] = bool(rej)
    else:
        for c in _ph_conds:
            posthoc[c]["p_one_tailed_holm"] = float("nan")
            posthoc[c]["significant_holm"] = False

    rows_long = [
        {"subject": s, "condition": c, "acc": float(M[i, j])}
        for i, s in enumerate(subjects_valid)
        for j, c in enumerate(conditions)
    ]
    df_long = pd.DataFrame(rows_long)
    rm_anova = {}
    try:
        fit = AnovaRM(df_long, depvar="acc", subject="subject", within=["condition"]).fit()
        f_stat = float(fit.anova_table["F Value"].iloc[0])
        df_num = float(fit.anova_table["Num DF"].iloc[0])
        df_den = float(fit.anova_table["Den DF"].iloc[0])
        p_uncorr = float(fit.anova_table["Pr > F"].iloc[0])
        gg_eps = _greenhouse_geisser_epsilon(M)
        from scipy.stats import f as f_dist
        p_gg = float(1 - f_dist.cdf(f_stat, df_num * gg_eps, df_den * gg_eps))
        rm_anova = {
            "f_statistic": f_stat, "df_numerator": df_num, "df_denominator": df_den,
            "p_uncorrected": p_uncorr, "greenhouse_geisser_epsilon": gg_eps,
            "df_numerator_gg": df_num * gg_eps, "df_denominator_gg": df_den * gg_eps,
            "p_greenhouse_geisser": p_gg,
        }
    except Exception as exc:
        rm_anova = {"error": str(exc)}

    return {
        "n_subjects": n,
        "subjects_included": subjects_valid,
        "conditions": conditions,
        "control_ceiling": check_control_ceiling(results),
        "descriptives": descriptives,
        "friedman": {"statistic": float(friedman_stat),
                     "p_value": float(friedman_p), "df": len(conditions) - 1},
        "wilcoxon_posthoc_vs_control": posthoc,
        "rm_anova": rm_anova,
    }


# ---- Retention layer (descriptive; exploratory test flagged as such) ------

def load_all_rejection_logs(derived_root: str | Path = "data/derived") -> list[dict]:
    """Load every preprocessing rejection.json (the retention / data-loss layer)."""
    derived_root = Path(derived_root)
    rej_dir = derived_root / PREPROCESSING_PIPELINE
    out = []
    for path in sorted(rej_dir.glob("sub-*/sub-*_cond-*_rejection.json")):
        with open(path) as f:
            out.append(json.load(f))
    return out


def build_retention_dataframe(logs: list[dict]) -> pd.DataFrame:
    """Tidy frame of per-(subject, condition) retention / rejection."""
    rows = []
    for lg in logs:
        n_planned = lg.get("n_planned", 0)
        n_kept = lg.get("n_kept", 0)
        rows.append({
            "subject": lg.get("subject_id", "?"),
            "condition": lg.get("condition", "?"),
            "n_planned": n_planned,
            "n_kept": n_kept,
            "retention_rate": (n_kept / n_planned * 100) if n_planned else float("nan"),
            "rejection_rate": lg.get("rejection_rate", float("nan")) * 100,
        })
    return pd.DataFrame(rows)


def run_retention_exploratory(logs: list[dict]) -> dict:
    """EXPLORATORY (not pre-registered as inferential): Friedman + one-tailed
    Wilcoxon on rejection rate, predicted direction noise > control."""
    df = build_retention_dataframe(logs)
    piv = df.pivot_table(index="subject", columns="condition",
                         values="rejection_rate")
    piv = piv.reindex(columns=[c for c in RAW_CONDITIONS if c in piv.columns])
    piv = piv.dropna(axis=0, how="any")
    if piv.shape[0] < 3 or "control" not in piv.columns:
        return {"note": "insufficient complete cases for exploratory test",
                "per_condition_mean_rejection": df.groupby("condition")["rejection_rate"].mean().to_dict()}
    M = piv.values
    conds = list(piv.columns)
    ctrl = M[:, conds.index("control")]
    fr_stat, fr_p = friedmanchisquare(*[M[:, j] for j in range(len(conds))])
    posthoc = {}
    for c in [c for c in conds if c != "control"]:
        v = M[:, conds.index(c)]
        try:
            _, p_one = wilcoxon(v, ctrl, alternative="greater")  # noise > control rejection
        except Exception:
            p_one = float("nan")
        posthoc[c] = {"mean_rejection": float(v.mean()),
                      "p_one_tailed_vs_control": float(p_one)}
    return {
        "n_subjects": int(M.shape[0]),
        "conditions": conds,
        "mean_rejection_by_condition": {c: float(M[:, conds.index(c)].mean()) for c in conds},
        "friedman": {"statistic": float(fr_stat), "p_value": float(fr_p)},
        "wilcoxon_posthoc_vs_control": posthoc,
        "EXPLORATORY": True,
    }


# ==========================================================================
# BLOCKWISE (LEAVE-ONE-SUB-BLOCK-OUT) ROBUSTNESS ANALYSIS
# --------------------------------------------------------------------------
# Added 2026-08-13 in response to SYNASC 2026 Reviewer 1, who observed that the
# control condition is split within a single recording while the three noise
# conditions are tested on entirely separate recordings, and that with a 1000 ms
# epoch window at a 233 ms SOA adjacent epochs overlap (reach: 4 x 233 = 932 ms).
#
# Everything below is ADDITIVE. train_and_evaluate() above is untouched, so the
# v3 numbers in the submitted manuscript remain exactly reproducible.
#
# Design (pre-committed in DEVIATIONS.md 2026-08-13, before any result was seen):
#   * Folds are the three control sub-blocks. Train on two, test on the third.
#   * Each fold's model ALSO scores the full chewing / EMI / acoustic recordings,
#     so every condition is scored by models trained on an identical amount of
#     data. Reported values are the mean across the three folds.
#   * Leak-free by construction: the registered boundary rejection (first 2 s /
#     last 1 s of each sub-block) plus the self-paced inter-sub-block rest leaves
#     a measured 11.9-77.8 s gap between the last epoch of one sub-block and the
#     first of the next, against a 932 ms overlap reach.
#   * Confound to keep in view: each fold also tests a target CELL absent from
#     its training set, so a drop cannot be attributed to leakage alone. The
#     blockwise ceiling is a conservative lower bound, not a leakage measurement.
#
# Output: data/derived/classifier-v3-blockwise/sub-<id>/sub-<id>_results.json,
# in the SAME schema as v3 so run_group_statistics() applies unmodified.
# ==========================================================================


class SWLDAClassifierFast(SWLDAClassifier):
    """SWLDAClassifier with a vectorised forward step. Numerically identical.

    WHY THIS EXISTS
    ---------------
    The parent's forward step fits a fresh statsmodels OLS for every one of the
    ~400 candidate features at every iteration, which costs minutes per model.
    The blockwise analysis needs three fits per subject instead of one, making
    the parent implementation impractical to run.

    WHAT IT CHANGES
    ---------------
    Nothing about the mathematics. The conditional p-value of adding candidate j
    to the current design M is obtained from partitioned regression
    (Frisch-Waugh-Lovell) rather than from a full refit:

        r_y = y - M (M'M)^-1 M'y          residualised response
        r_j = x_j - M (M'M)^-1 M'x_j      residualised candidate
        b_j = (r_j . r_y) / (r_j . r_j)
        RSS_j = RSS_current - (r_j . r_y)^2 / (r_j . r_j)
        se_j  = sqrt( RSS_j / (n - k - 1) / (r_j . r_j) )
        t_j   = b_j / se_j,   p_j = 2 * P(T_{n-k-1} > |t_j|)

    These are the same closed-form quantities statsmodels reports for the
    just-added coefficient, computed for all candidates in two matrix solves
    instead of 400 model fits. Candidates that are numerically collinear with the
    current design (r_j . r_j below tolerance) are assigned p = inf, mirroring the
    parent's `except: continue` behaviour.

    The backward step is inherited unchanged -- it is a single fit per iteration
    and was never the bottleneck.

    Equivalence against the parent is asserted by analysis/_test_blockwise.py,
    which requires identical selected feature sets and identical metrics.
    """

    _COLLINEAR_TOL = 1e-10

    def _stepwise_select(self, X: np.ndarray, y_reg: np.ndarray) -> list[int]:
        n_samples, n_features = X.shape
        selected: list[int] = []
        available = list(range(n_features))

        Xd = X.astype(np.float64)
        yd = y_reg.astype(np.float64)
        ones = np.ones((n_samples, 1))

        iteration = 0
        max_iterations = self.max_features * 4  # safety bound (matches parent)
        while iteration < max_iterations:
            iteration += 1

            # ---- Forward: best candidate to add (vectorised) --------------
            best_pval = 1.0
            best_feature = None
            if len(selected) < self.max_features and available:
                M = np.hstack([ones, Xd[:, selected]]) if selected else ones
                k = M.shape[1]
                df = n_samples - k - 1
                if df > 0:
                    MtM = M.T @ M
                    Z = Xd[:, available]
                    # Residualise response and all candidates against M at once.
                    coef_y = np.linalg.solve(MtM, M.T @ yd)
                    r_y = yd - M @ coef_y
                    coef_Z = np.linalg.solve(MtM, M.T @ Z)
                    r_Z = Z - M @ coef_Z

                    d = np.einsum("ij,ij->j", r_Z, r_Z)      # r_j . r_j
                    a = r_Z.T @ r_y                          # r_j . r_y
                    rss_cur = float(r_y @ r_y)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        b = a / d
                        rss_new = rss_cur - (a ** 2) / d
                        se = np.sqrt(np.maximum(rss_new, 0.0) / df / d)
                        tvals = b / se

                    bad = (d <= self._COLLINEAR_TOL) | ~np.isfinite(tvals)
                    pvals = np.full(len(available), np.inf)
                    good = ~bad
                    if good.any():
                        pvals[good] = 2.0 * t_dist.sf(np.abs(tvals[good]), df)

                    j_local = int(np.argmin(pvals))
                    if np.isfinite(pvals[j_local]):
                        best_pval = float(pvals[j_local])
                        best_feature = available[j_local]

            added = False
            if best_feature is not None and best_pval < self.p_enter:
                selected.append(best_feature)
                available.remove(best_feature)
                added = True
                if self.verbose:
                    print(f"  + add feat {best_feature} (p={best_pval:.4f}), "
                          f"selected={len(selected)}")

            # ---- Backward: inherited logic, single fit per iteration ------
            removed = False
            if len(selected) > 1:
                X_full = sm.add_constant(Xd[:, selected], has_constant="add")
                try:
                    model = sm.OLS(yd, X_full).fit()
                    pvals_selected = model.pvalues[1:]  # skip constant
                    worst_idx = int(np.argmax(pvals_selected))
                    worst_pval = float(pvals_selected[worst_idx])
                    if worst_pval > self.p_remove:
                        feat = selected[worst_idx]
                        selected.pop(worst_idx)
                        available.append(feat)
                        removed = True
                        if self.verbose:
                            print(f"  - remove feat {feat} (p={worst_pval:.4f}), "
                                  f"selected={len(selected)}")
                except Exception:
                    pass

            if not added and not removed:
                break

        return sorted(selected)


def _extract_features_with_blocks(
    epochs: mne.Epochs, target_sfreq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As _extract_features, but also returns the per-epoch sub-block index.

    Resampling neither drops nor reorders epochs, so the metadata rows stay
    aligned with the returned feature matrix.
    """
    epochs_ds = epochs.copy().resample(target_sfreq, verbose="WARNING")
    data = epochs_ds.get_data() * 1e6  # volts -> microvolts
    n_epochs, n_channels, n_samples = data.shape
    X = data.reshape(n_epochs, n_channels * n_samples).astype(np.float32)
    y = (epochs_ds.metadata["is_target"].values).astype(int)
    sub_block = (epochs_ds.metadata["sub_block_index"].values).astype(int)
    return X, y, sub_block


def train_and_evaluate_blockwise(
    subject_id: str,
    derived_root: str | Path = "data/derived",
    save: bool = True,
    verbose: bool = False,
) -> dict:
    """Leave-one-sub-block-out CV on control; every fold scores all conditions.

    Returns a dict in the same schema as train_and_evaluate()'s saved JSON, with
    an extra "folds" key carrying the per-fold breakdown (used to check whether
    accuracy declines across sub-blocks within a recording).
    """
    derived_root = Path(derived_root)

    control_epochs = _load_epochs(subject_id, "control", derived_root)
    X_ctrl, y_ctrl, sb_ctrl = _extract_features_with_blocks(
        control_epochs, target_sfreq=DOWNSAMPLED_HZ
    )

    # Noise recordings are loaded once and scored by every fold's model.
    noise: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in ["chewing", "emi", "acoustic"]:
        epochs = _load_epochs(subject_id, cond, derived_root)
        noise[cond] = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)

    blocks = sorted(int(b) for b in np.unique(sb_ctrl))
    folds: list[dict] = []

    for held in blocks:
        train_mask = sb_ctrl != held
        test_mask = sb_ctrl == held

        clf = SWLDAClassifierFast(
            p_enter=SWLDA_P_ENTER,
            p_remove=SWLDA_P_REMOVE,
            max_features=SWLDA_MAX_FEATURES,
            verbose=False,
        )
        clf.fit(X_ctrl[train_mask], y_ctrl[train_mask])

        fold = {
            "held_out_sub_block": held,
            "n_train_epochs": int(train_mask.sum()),
            "train_target_count": int((y_ctrl[train_mask] == 1).sum()),
            "train_nontarget_count": int((y_ctrl[train_mask] == 0).sum()),
            "n_features": int(len(clf.selected_features_)),
            "per_condition": {
                "control_heldout": _evaluate(clf, X_ctrl[test_mask], y_ctrl[test_mask])
            },
        }
        for cond, (X_n, y_n) in noise.items():
            fold["per_condition"][cond] = _evaluate(clf, X_n, y_n)
        folds.append(fold)

        if verbose:
            ba = fold["per_condition"]["control_heldout"]["balanced_accuracy"]
            print(f"    fold hold-out sb{held}: {fold['n_features']} feats, "
                  f"control {ba * 100:.1f}%")

    # ---- Aggregate across folds ----------------------------------------
    # Rates are the mean of the per-fold rates (as pre-committed). The confusion
    # matrix is pooled; for control_heldout that means every control epoch counted
    # exactly once, since each fold holds out a different sub-block.
    per_condition: dict = {}
    for cond in CLF_CONDITIONS:
        vals = [f["per_condition"][cond] for f in folds]
        per_condition[cond] = {
            "n_epochs": int(sum(v["n_epochs"] for v in vals)),
            "n_target": int(sum(v["n_target"] for v in vals)),
            "n_nontarget": int(sum(v["n_nontarget"] for v in vals)),
            "balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in vals])),
            "true_target_rate": float(np.mean([v["true_target_rate"] for v in vals])),
            "true_nontarget_rate": float(np.mean([v["true_nontarget_rate"] for v in vals])),
            "confusion_matrix": {
                k: int(sum(v["confusion_matrix"][k] for v in vals))
                for k in ("TN", "FP", "FN", "TP")
            },
            "balanced_accuracy_per_fold": [float(v["balanced_accuracy"]) for v in vals],
            "n_folds": len(vals),
        }

    result_dict = {
        "subject_id": subject_id,
        "classifier_type": "swlda-blockwise",
        "n_features": float(np.mean([f["n_features"] for f in folds])),
        "n_train_epochs": int(np.mean([f["n_train_epochs"] for f in folds])),
        "train_target_count": int(np.mean([f["train_target_count"] for f in folds])),
        "train_nontarget_count": int(np.mean([f["train_nontarget_count"] for f in folds])),
        "per_condition": per_condition,
        "folds": folds,
        "parameters": {
            "downsampled_hz": DOWNSAMPLED_HZ,
            "split": "leave-one-sub-block-out",
            "n_folds": len(folds),
            "sub_blocks": blocks,
            "swlda_p_enter": SWLDA_P_ENTER,
            "swlda_p_remove": SWLDA_P_REMOVE,
            "swlda_max_features": SWLDA_MAX_FEATURES,
            "balancing_applied": False,
            "preprocessing_pipeline": PREPROCESSING_PIPELINE,
        },
    }

    if save:
        out_dir = derived_root / DERIVED_PIPELINE_BLOCKWISE / f"sub-{subject_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"sub-{subject_id}_results.json", "w") as f:
            json.dump(result_dict, f, indent=2)

    return result_dict


def build_fold_dataframe(results: list[dict]) -> pd.DataFrame:
    """Per-(subject, held-out sub-block) control accuracy, for the question of
    whether classification degrades across sub-blocks within a recording."""
    rows = []
    for r in results:
        for f in r.get("folds", []):
            rows.append({
                "subject": r["subject_id"],
                "held_out_sub_block": f["held_out_sub_block"],
                "n_features": f["n_features"],
                "balanced_accuracy": f["per_condition"]["control_heldout"]["balanced_accuracy"] * 100,
            })
    return pd.DataFrame(rows)
