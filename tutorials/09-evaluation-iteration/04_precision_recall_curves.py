# Problem 4: Precision-Recall Curves
#
# Implement precision-recall curves for imbalanced classification problems
# where ROC can be misleading.
#
# Example:
#   precision, recall, thresholds = compute_pr_curve(y_true, y_scores)
#   ap = average_precision(y_true, y_scores)
#   # Returns: 0.72 (average precision)
#
# ML Relevance: PR curves are preferred when:
# - Classes are highly imbalanced
# - False positives are costly
# - You care about positive class performance

from typing import List, Dict, Tuple
import numpy as np


def compute_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.

    Returns:
        (precision, recall, thresholds) arrays
    """
    # Your solution here
    pass


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute average precision (area under PR curve).

    Returns:
        AP score
    """
    # Your solution here
    pass


def f1_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> float:
    """Compute F1 at a specific threshold."""
    # Your solution here
    pass


def find_threshold_for_precision(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    min_precision: float
) -> Tuple[float, float]:
    """
    Find threshold that achieves minimum precision.

    Returns:
        (threshold, corresponding recall)
    """
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # Imbalanced
    y_scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 0.8, 0.9])

    print("Testing Precision-Recall curves...")

    # Test 1: PR curve
    prec, rec, thresh = compute_pr_curve(y_true, y_scores)
    assert len(prec) == len(rec), "Test 1 failed"
    print(f"  ✓ PR curve: {len(prec)} points")

    # Test 2: Average precision
    ap = average_precision(y_true, y_scores)
    assert 0 <= ap <= 1, f"Test 2 failed: {ap}"
    print(f"  ✓ Average Precision: {ap:.4f}")

    # Test 3: F1 at threshold
    f1 = f1_at_threshold(y_true, y_scores, 0.5)
    assert 0 <= f1 <= 1, f"Test 3 failed: {f1}"
    print(f"  ✓ F1 at 0.5: {f1:.4f}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
