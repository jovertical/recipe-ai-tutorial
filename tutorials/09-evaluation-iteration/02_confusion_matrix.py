# Problem 2: Confusion Matrix Analysis
#
# Build and analyze confusion matrices to understand model errors in detail.
#
# Example:
#   cm = ConfusionMatrix(y_true, y_pred, labels=["cat", "dog", "bird"])
#   cm.plot()
#   errors = cm.get_common_errors()
#   # Returns: [("cat", "dog", 15), ("bird", "cat", 8), ...]
#
# ML Relevance: Confusion matrices reveal:
# - Which classes are confused with each other
# - Class imbalance effects
# - Systematic prediction biases
# - Where to focus improvement efforts

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ConfusionMatrixResult:
    """Confusion matrix with metadata."""
    matrix: np.ndarray
    labels: List[str]
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = None
) -> np.ndarray:
    """
    Build confusion matrix from predictions.

    Args:
        y_true: Ground truth labels (integers)
        y_pred: Predicted labels (integers)
        num_classes: Number of classes (inferred if None)

    Returns:
        Confusion matrix (num_classes x num_classes)
        Row i, Column j = count of true class i predicted as j
    """
    # Your solution here
    pass


def normalize_confusion_matrix(
    cm: np.ndarray,
    mode: str = "true"
) -> np.ndarray:
    """
    Normalize confusion matrix.

    Args:
        cm: Raw confusion matrix
        mode: "true" (by true labels), "pred" (by predictions), "all" (total)

    Returns:
        Normalized confusion matrix
    """
    # Your solution here
    pass


class ConfusionMatrix:
    """Confusion matrix analyzer."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None
    ):
        """Initialize with predictions."""
        # Your solution here
        pass

    def get_common_errors(self, top_k: int = 5) -> List[Tuple[str, str, int]]:
        """
        Get most common prediction errors.

        Returns:
            List of (true_label, pred_label, count) sorted by count
        """
        # Your solution here
        pass

    def class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, F1.

        Returns:
            {class_name: {"precision": x, "recall": y, "f1": z}}
        """
        # Your solution here
        pass

    def to_string(self) -> str:
        """Pretty-print the confusion matrix."""
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 0])
    labels = ["cat", "dog", "bird"]

    print("Testing confusion matrix...")

    # Test 1: Build matrix
    cm = build_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3), f"Test 1 failed: {cm.shape}"
    print(f"  ✓ Matrix shape: {cm.shape}")

    # Test 2: Normalize
    cm_norm = normalize_confusion_matrix(cm, mode="true")
    assert np.allclose(cm_norm.sum(axis=1), 1.0), "Test 2 failed"
    print(f"  ✓ Normalized matrix rows sum to 1")

    # Test 3: ConfusionMatrix class
    analyzer = ConfusionMatrix(y_true, y_pred, labels=labels)
    print(f"  ✓ ConfusionMatrix initialized")

    # Test 4: Common errors
    errors = analyzer.get_common_errors(top_k=3)
    assert len(errors) <= 3, f"Test 4 failed: {len(errors)}"
    print(f"  ✓ Common errors: {errors}")

    # Test 5: Class metrics
    metrics = analyzer.class_metrics()
    assert "cat" in metrics, "Test 5 failed"
    print(f"  ✓ Class metrics: {list(metrics.keys())}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
