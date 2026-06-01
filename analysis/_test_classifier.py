"""Smoke test for the classifier scaffold using LDA.

Trains on control, evaluates on all four conditions. Prints per-condition
balanced accuracy and confusion matrix breakdown.
"""
from analysis.classifier import train_and_evaluate


def main():
    subject = "pilot-self-day0"
    print(f"\n=== TRAINING (LDA, scaffold validation) ===")
    print(f"  subject: {subject}")
    print(f"  classifier: LDA with shrinkage=auto (scaffold)")

    result = train_and_evaluate(subject_id=subject, classifier_type="swlda")

    print(f"\n  Features per epoch:  {result.n_features}")
    print(f"  Train epochs:        {result.n_train_epochs}")
    print(f"    targets:           {result.train_target_count}")
    print(f"    nontargets:        {result.train_nontarget_count}")

    print(f"\n=== PER-CONDITION RESULTS ===")
    print(f"  {'condition':<20} {'bal_acc':>8} {'true_tgt':>10} {'true_nontgt':>12} "
          f"{'n_tgt':>7} {'n_nontgt':>9}")
    print(f"  {'-'*70}")
    for cond, metrics in result.per_condition.items():
        print(f"  {cond:<20} "
              f"{metrics['balanced_accuracy']*100:>7.1f}% "
              f"{metrics['true_target_rate']*100:>9.1f}% "
              f"{metrics['true_nontarget_rate']*100:>11.1f}% "
              f"{metrics['n_target']:>7} "
              f"{metrics['n_nontarget']:>9}")

        cm = metrics['confusion_matrix']
        print(f"    Raw CM -> True Pos: {cm['TP']:<4} False Neg: {cm['FN']:<4} | "
              f"True Neg: {cm['TN']:<4} False Pos: {cm['FP']:<4}")

    print(f"\n=== INTERPRETATION ===")
    ctrl_acc = result.per_condition["control_heldout"]["balanced_accuracy"]
    print(f"  Control held-out bal_acc: {ctrl_acc*100:.1f}%")
    if ctrl_acc < 0.60:
        print(f"  ⚠  Below 60% threshold — pre-reg §5 contingency applies if this")
        print(f"     pattern holds across real subjects (N170 becomes primary).")
    else:
        print(f"  ✓  Above 60% threshold — primary analysis is viable.")

    for cond in ["chewing", "emi", "acoustic"]:
        delta = (result.per_condition[cond]["balanced_accuracy"]
                 - ctrl_acc) * 100
        direction = "↓" if delta < 0 else "↑"
        print(f"  {cond:>10}: Δ from control = {direction} {abs(delta):>5.1f} percentage points")


if __name__ == "__main__":
    main()
