"""Analyze ∆E (spilled energy) from greedy eval results.

Computes AUROC and optimal threshold for hallucination detection
using the paper's approach: min/max/mean pooling of per-token ∆E
over the generated answer.

Usage:
    python -m evals.analyze_delta_e results/capitals_qwen35_0b8_greedy.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def analyze(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    correct, incorrect = [], []

    for r in results:
        delta_e = r.get("delta_e")
        if not delta_e or len(delta_e) < 2:
            continue
        # Skip step 0 (no previous logit → ∆E=0).
        values = np.array(delta_e[1:])
        entry = {
            "case": r["case"],
            "passed": r["passed"],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "values": values,
        }
        if r["passed"]:
            correct.append(entry)
        else:
            incorrect.append(entry)

    if not correct or not incorrect:
        print(f"Need both correct and incorrect cases. Got {len(correct)} correct, {len(incorrect)} incorrect.")
        return

    print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")
    print()

    # Distribution summary.
    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        mins = [e["min"] for e in group]
        maxs = [e["max"] for e in group]
        means = [e["mean"] for e in group]
        print(f"{label}:")
        print(f"  ∆E min  — median={np.median(mins):.3f}  mean={np.mean(mins):.3f}  std={np.std(mins):.3f}")
        print(f"  ∆E max  — median={np.median(maxs):.3f}  mean={np.mean(maxs):.3f}  std={np.std(maxs):.3f}")
        print(f"  ∆E mean — median={np.median(means):.3f}  mean={np.mean(means):.3f}  std={np.std(means):.3f}")
        print()

    # AUROC for each pooling strategy.
    # Convention: higher score → more likely hallucination (incorrect).
    # Labels: 1 = incorrect (hallucination), 0 = correct.
    all_entries = correct + incorrect
    labels = np.array([0] * len(correct) + [1] * len(incorrect))

    for strategy in ["min", "max", "mean"]:
        scores = np.array([e[strategy] for e in all_entries])
        auroc = _auroc(labels, scores)

        # Also try negative scores (lower ∆E → hallucination).
        auroc_neg = _auroc(labels, -scores)

        best_auroc = max(auroc, auroc_neg)
        direction = "higher" if auroc >= auroc_neg else "lower"

        # Find optimal threshold (maximize Youden's J = TPR - FPR).
        if direction == "higher":
            threshold, tpr, fpr, f1 = _optimal_threshold(labels, scores)
        else:
            threshold, tpr, fpr, f1 = _optimal_threshold(labels, -scores)
            threshold = -threshold  # flip back to original scale

        print(f"Pool={strategy:4s}  AUROC={best_auroc:.3f}  direction={direction}  "
              f"threshold={threshold:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}  F1={f1:.3f}")

    print()
    # Show per-case details for cases near the decision boundary.
    print("--- Per-case ∆E (sorted by min ∆E) ---")
    all_sorted = sorted(all_entries, key=lambda e: e["min"])
    for e in all_sorted:
        mark = "✓" if e["passed"] else "✗"
        print(f"  {mark} {e['case']:40s}  min={e['min']:+.3f}  max={e['max']:+.3f}  mean={e['mean']:+.3f}")


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney statistic.
    count = 0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))


def _optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Find threshold maximizing Youden's J = TPR - FPR. Returns (threshold, tpr, fpr, f1)."""
    thresholds = np.sort(np.unique(scores))
    best_j, best_t, best_tpr, best_fpr = -1.0, 0.0, 0.0, 1.0
    for t in thresholds:
        predicted = (scores >= t).astype(int)
        tp = np.sum((predicted == 1) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        fn = np.sum((predicted == 0) & (labels == 1))
        tn = np.sum((predicted == 0) & (labels == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
            best_tpr = tpr
            best_fpr = fpr
    # Compute F1 at best threshold.
    predicted = (scores >= best_t).astype(int)
    tp = np.sum((predicted == 1) & (labels == 1))
    fp = np.sum((predicted == 1) & (labels == 0))
    fn = np.sum((predicted == 0) & (labels == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(best_t), float(best_tpr), float(best_fpr), float(f1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evals.analyze_delta_e <results.json>")
        sys.exit(1)
    analyze(sys.argv[1])
