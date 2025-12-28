# Problem 3: ROC Curves and AUC
#
# Implement ROC curve computation and AUC scoring for threshold-independent
# evaluation of binary classifiers.
#
# Example:
#   fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
#   auc = compute_auc(fpr, tpr)
#   # Returns: 0.87 (good discrimination)
#
# ML Relevance: ROC/AUC is essential for:
# - Comparing models regardless of threshold choice
# - Evaluating ranking quality
# - Understanding sensitivity vs specificity tradeoffs
# - Medical/fraud detection where thresholds matter

from typing import List, Dict, Tuple
import numpy as np


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve points.

    Args:
        y_true: Binary ground truth (0 or 1)
        y_scores: Predicted probabilities for positive class
        num_thresholds: Number of threshold points

    Returns:
        (fpr, tpr, thresholds) arrays
    """
    # Your solution here
    pass


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve using trapezoidal rule.

    Args:
        fpr: False positive rates
        tpr: True positive rates

    Returns:
        AUC score (0 to 1)
    """
    # Your solution here
    pass


def find_optimal_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    method: str = "youden"
) -> float:
    """
    Find optimal classification threshold.

    Args:
        fpr, tpr, thresholds: From compute_roc_curve
        method: "youden" (max TPR-FPR) or "closest" (closest to (0,1))

    Returns:
        Optimal threshold
    """
    # Your solution here
    pass


class ROCAnalyzer:
    """ROC curve analysis tools."""

    def __init__(self, y_true: np.ndarray, y_scores: np.ndarray):
        """Initialize with predictions."""
        # Your solution here
        pass

    def auc(self) -> float:
        """Return AUC score."""
        # Your solution here
        pass

    def optimal_threshold(self, method: str = "youden") -> float:
        """Find optimal threshold."""
        # Your solution here
        pass

    def partial_auc(self, fpr_range: Tuple[float, float]) -> float:
        """
        Compute partial AUC within FPR range.

        Useful when only low FPR region matters.
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    # Good classifier
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9])

    print("Testing ROC/AUC...")

    # Test 1: ROC curve
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    assert fpr[0] == 0 and fpr[-1] == 1, "Test 1a failed"
    assert tpr[0] == 0 and tpr[-1] == 1, "Test 1b failed"
    print(f"  ✓ ROC curve: {len(fpr)} points")

    # Test 2: AUC
    auc = compute_auc(fpr, tpr)
    assert 0.8 <= auc <= 1.0, f"Test 2 failed: {auc}"
    print(f"  ✓ AUC: {auc:.4f}")

    # Test 3: Optimal threshold
    thresh = find_optimal_threshold(fpr, tpr, thresholds)
    assert 0 < thresh < 1, f"Test 3 failed: {thresh}"
    print(f"  ✓ Optimal threshold: {thresh:.4f}")

    # Test 4: ROCAnalyzer
    analyzer = ROCAnalyzer(y_true, y_scores)
    assert analyzer.auc() > 0.8, "Test 4 failed"
    print(f"  ✓ ROCAnalyzer AUC: {analyzer.auc():.4f}")

    # Test 5: Random classifier should have AUC ~0.5
    y_random = np.random.rand(100)
    y_true_rand = np.random.randint(0, 2, 100)
    fpr_r, tpr_r, _ = compute_roc_curve(y_true_rand, y_random)
    auc_random = compute_auc(fpr_r, tpr_r)
    assert 0.3 <= auc_random <= 0.7, f"Test 5 failed: {auc_random}"
    print(f"  ✓ Random AUC: {auc_random:.4f} (expected ~0.5)")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
