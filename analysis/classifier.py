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
are regenerated with v3. One must regenerate everything if they previously used v2.

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import statsmodels.api as sm


# ---- Pre-registered parameters ----------------------------------------
# §5: 50 samples per epoch (1 per 20 ms over the 1000 ms window).
DOWNSAMPLED_HZ = 50
# §5: 70/30 train/test split on control for the within-condition ceiling.
TRAIN_FRACTION = 0.70

DERIVED_PIPELINE = "classifier-v3"
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
