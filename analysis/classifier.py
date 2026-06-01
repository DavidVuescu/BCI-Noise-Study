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


# ---- Pre-registered parameters ----------------------------------------
# §5: 50 samples per epoch (1 per 20 ms over the 1000 ms window).
DOWNSAMPLED_HZ = 50
# §5: 70/30 train/test split on control for the within-condition ceiling.
TRAIN_FRACTION = 0.70

DERIVED_PIPELINE = "classifier-v1"
PREPROCESSING_PIPELINE = "preprocessing-v1"


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

    if classifier_type == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier_type == "swlda":
        raise NotImplementedError("SWLDA added after LDA scaffold validates")
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    clf.fit(X_train, y_train)

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
        },
    )

    # ---- Evaluate: held-out control + each noise condition -------------
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
