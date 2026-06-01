"""
Classifier pipeline: trains a per-subject classifier on control epochs,
evaluates on all four conditions.

Scaffold uses plain LDA for initial validation. SWLDA (stepwise feature
selection wrapper) is added as a separate function once the scaffold is
trusted; per pre-reg §5, SWLDA is the final classifier.

Output paths:
    data/derived/classifier-v1/sub-<id>/sub-<id>_model.pkl
    data/derived/classifier-v1/sub-<id>/sub-<id>_results.json
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

DERIVED_PIPELINE = "classifier-v1"
PREPROCESSING_PIPELINE = "preprocessing-v1"



# ---- SWLDA parameters (Krusienski et al. 2008, canonical for P300 spellers) ---
# Locked before any data is collected. Changing requires deviation logging.
SWLDA_P_ENTER = 0.1     # feature added if conditional p-value < this
SWLDA_P_REMOVE = 0.15   # feature removed if conditional p-value > this
SWLDA_MAX_FEATURES = 60 # hard cap on selected feature count


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
    # epochs_ds.get_data() returns (n_epochs, n_channels, n_samples) in volts.
    # Scale back to microvolts so feature magnitudes are reasonable (~1-100
    # instead of ~1e-6 to 1e-4) — helps numerical stability of LDA's covariance.
    data = epochs_ds.get_data() * 1e6
    n_epochs, n_channels, n_samples = data.shape
    X = data.reshape(n_epochs, n_channels * n_samples).astype(np.float32)
    y = (epochs_ds.metadata["is_target"].values).astype(int)
    return X, y


def _swlda_select_features(
    X: np.ndarray,
    y: np.ndarray,
    p_enter: float = SWLDA_P_ENTER,
    p_remove: float = SWLDA_P_REMOVE,
    max_features: int = SWLDA_MAX_FEATURES,
    verbose: bool = False,
) -> list[int]:
    """Stepwise feature selection for SWLDA, per Krusienski et al. 2008.

    Uses OLS regression with the binary label as the dependent variable.
    Forward selection by best conditional p-value, then backward elimination
    of features whose conditional p-value drifts above p_remove after other
    features are added.

    Returns:
        Sorted list of selected feature column indices.
    """
    n_samples, n_features = X.shape
    selected: list[int] = []
    available = list(range(n_features))

    # Center y to mimic the regression-on-residuals setup; OLS handles this
    # but explicit centering makes the linear-algebra equivalent clearer.
    y_float = y.astype(float)

    iteration = 0
    max_iterations = max_features * 4  # safety bound, shouldn't be hit
    while iteration < max_iterations:
        iteration += 1

        # ---- Forward step: find best candidate to add --------------------
        best_pval = 1.0
        best_feature = None

        if len(selected) < max_features:
            for j in available:
                trial = selected + [j]
                X_trial = sm.add_constant(X[:, trial], has_constant="add")
                try:
                    model = sm.OLS(y_float, X_trial).fit()
                    # p-value of the newly-added feature is the last coefficient
                    # after the constant; coefficient index = len(trial)
                    pval = model.pvalues[-1]
                except Exception:
                    continue
                if pval < best_pval:
                    best_pval = pval
                    best_feature = j

        added = False
        if best_feature is not None and best_pval < p_enter:
            selected.append(best_feature)
            available.remove(best_feature)
            added = True
            if verbose:
                print(f"  + add feat {best_feature} (p={best_pval:.4f}), "
                      f"selected={len(selected)}")

        # ---- Backward step: remove any feature whose p-value drifted up --
        removed = False
        if len(selected) > 1:
            X_full = sm.add_constant(X[:, selected], has_constant="add")
            try:
                model = sm.OLS(y_float, X_full).fit()
                # Skip the constant (index 0); check selected features
                pvals_selected = model.pvalues[1:]
                worst_idx = int(np.argmax(pvals_selected))
                worst_pval = float(pvals_selected[worst_idx])
                if worst_pval > p_remove:
                    feat_to_remove = selected[worst_idx]
                    selected.pop(worst_idx)
                    available.append(feat_to_remove)
                    removed = True
                    if verbose:
                        print(f"  - remove feat {feat_to_remove} "
                              f"(p={worst_pval:.4f}), selected={len(selected)}")
            except Exception:
                pass

        # Stop when neither forward nor backward step changed anything
        if not added and not removed:
            break

    return sorted(selected)


def _evaluate(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Run prediction and compute the suite of metrics we care about."""
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # Confusion matrix layout: rows = true, cols = predicted
    #   [[TN, FP],
    #    [FN, TP]]
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


def train_and_evaluate(
    subject_id: str,
    classifier_type: str = "lda",
    derived_root: str | Path = "data/derived",
    save: bool = True,
    random_state: int = 42,
) -> ClassifierResult:
    """Train classifier on control, evaluate on all four conditions.

    Args:
        subject_id: e.g. "pilot-self-day0"
        classifier_type: "lda" for scaffold validation, "swlda" for the
            registered final classifier (not yet implemented in this file).
        derived_root: base path for preprocessing inputs and classifier outputs.
        save: persist model and results JSON.
        random_state: for the train/test split reproducibility.

    Returns:
        ClassifierResult with per-condition metrics.
    """
    derived_root = Path(derived_root)

    # ---- Load control, split, train ------------------------------------
    control_epochs = _load_epochs(subject_id, "control", derived_root)
    X_ctrl, y_ctrl = _extract_features(control_epochs, target_sfreq=DOWNSAMPLED_HZ)

    X_train, X_test, y_train, y_test = train_test_split(
        X_ctrl, y_ctrl,
        train_size=TRAIN_FRACTION,
        stratify=y_ctrl,           # keep target/nontarget ratio in both splits
        random_state=random_state,
    )

    # ---- Class-balance the training set ----------------------------------
    # With ~1:7 target:nontarget imbalance, LDA defaults to predicting
    # nontarget always (which gets ~90% raw accuracy but 50% balanced).
    # Subsample the majority class to match the minority for training only.
    # Test sets are NOT rebalanced — we evaluate on the natural distribution.
    rng = np.random.default_rng(random_state)
    target_idx = np.where(y_train == 1)[0]
    nontarget_idx = np.where(y_train == 0)[0]
    n_min = len(target_idx)
    nontarget_kept = rng.choice(nontarget_idx, size=n_min, replace=False)
    train_idx = np.concatenate([target_idx, nontarget_kept])
    rng.shuffle(train_idx)
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    selected_features: list[int] | None = None
    if classifier_type == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_train, y_train)
    elif classifier_type == "swlda":
        # Step 1: stepwise feature selection on training data only
        selected_features = _swlda_select_features(
            X_train, y_train,
            p_enter=SWLDA_P_ENTER,
            p_remove=SWLDA_P_REMOVE,
            max_features=SWLDA_MAX_FEATURES,
            verbose=True,
        )
        if len(selected_features) == 0:
            raise RuntimeError("SWLDA selected zero features. "
                               "Signal is too weak for this configuration.")
        print(f"  SWLDA selected {len(selected_features)} of "
              f"{X_train.shape[1]} features")
        # Step 2: fit LDA on the selected features only
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    result = ClassifierResult(
        subject_id=subject_id,
        classifier_type=classifier_type,
        n_features=int(X_train.shape[1]),
        n_train_epochs=int(len(y_train)),
        train_target_count=int(y_train.sum()),
        train_nontarget_count=int((1 - y_train).sum()),
        parameters={
            "downsampled_hz": DOWNSAMPLED_HZ,
            "train_fraction": TRAIN_FRACTION,
            "shrinkage": "auto",
            "solver": "lsqr",
            "random_state": random_state,
            "swlda_p_enter": SWLDA_P_ENTER if classifier_type == "swlda" else None,
            "swlda_p_remove": SWLDA_P_REMOVE if classifier_type == "swlda" else None,
            "swlda_max_features": SWLDA_MAX_FEATURES if classifier_type == "swlda" else None,
            "swlda_selected_features": selected_features,
        },
    )

    # ---- Evaluate: held-out control + each noise condition -------------
    result.per_condition["control_heldout"] = _evaluate(clf, X_test, y_test)

    for cond in ["chewing", "emi", "acoustic"]:
        epochs = _load_epochs(subject_id, cond, derived_root)
        X, y = _extract_features(epochs, target_sfreq=DOWNSAMPLED_HZ)
        if selected_features is not None:
            X = X[:, selected_features]
        result.per_condition[cond] = _evaluate(clf, X, y)

    # ---- Persist -------------------------------------------------------
    if save:
        out_dir = derived_root / DERIVED_PIPELINE / f"sub-{subject_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"sub-{subject_id}_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        # Convert dataclass to dict for JSON
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
