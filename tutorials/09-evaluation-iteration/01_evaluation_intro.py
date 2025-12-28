# Problem 1: Introduction to Model Evaluation
#
# Understand the fundamentals of evaluating ML models - why accuracy alone
# is insufficient and how to design comprehensive evaluation strategies.
#
# Example:
#   evaluator = ModelEvaluator(model)
#   metrics = evaluator.evaluate(test_data)
#   # Returns: {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, ...}
#
# ML Relevance: Evaluation determines if your model is actually useful:
# - Identifies failure modes before production
# - Guides iteration and improvement
# - Builds confidence for deployment
# - Enables comparison between approaches

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    metric_name: str
    score: float
    confidence_interval: Tuple[float, float] = None
    metadata: Dict = None


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy: fraction of correct predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your solution here
    pass


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    """
    Compute precision: TP / (TP + FP).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Which class is "positive"

    Returns:
        Precision score
    """
    # Your solution here
    pass


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    """
    Compute recall: TP / (TP + FN).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Which class is "positive"

    Returns:
        Recall score
    """
    # Your solution here
    pass


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score: harmonic mean of precision and recall.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        F1 score
    """
    # Your solution here
    pass


class ModelEvaluator:
    """Comprehensive model evaluator."""

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator with specified metrics.

        Args:
            metrics: List of metric names to compute
        """
        # Your solution here
        pass

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Run all configured evaluations.

        Args:
            y_true: Ground truth
            y_pred: Predictions
            y_scores: Prediction probabilities (optional)

        Returns:
            Dictionary of metric names to scores
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    # Test data
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

    print("Testing evaluation metrics...")

    # Test 1: Accuracy
    acc = accuracy(y_true, y_pred)
    assert 0 <= acc <= 1, f"Test 1 failed: {acc}"
    print(f"  ✓ Accuracy: {acc:.4f}")

    # Test 2: Precision
    prec = precision(y_true, y_pred)
    assert 0 <= prec <= 1, f"Test 2 failed: {prec}"
    print(f"  ✓ Precision: {prec:.4f}")

    # Test 3: Recall
    rec = recall(y_true, y_pred)
    assert 0 <= rec <= 1, f"Test 3 failed: {rec}"
    print(f"  ✓ Recall: {rec:.4f}")

    # Test 4: F1
    f1 = f1_score(y_true, y_pred)
    assert 0 <= f1 <= 1, f"Test 4 failed: {f1}"
    print(f"  ✓ F1 Score: {f1:.4f}")

    # Test 5: Evaluator
    evaluator = ModelEvaluator(metrics=["accuracy", "precision", "recall", "f1"])
    results = evaluator.evaluate(y_true, y_pred)
    assert "accuracy" in results, "Test 5 failed"
    print(f"  ✓ Evaluator results: {results}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
