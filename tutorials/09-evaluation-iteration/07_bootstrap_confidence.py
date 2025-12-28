# Problem 7: Bootstrap Confidence Intervals
#
# Implement bootstrap resampling for confidence interval estimation.
#
# Example:
#   mean, ci_low, ci_high = bootstrap_confidence_interval(scores, metric_fn)
#   # Returns: (0.85, 0.82, 0.88) - mean and 95% CI
#
# ML Relevance: Confidence intervals:
# - Quantify uncertainty in metrics
# - Enable statistical significance testing
# - Help decide if improvements are real

from typing import Tuple, Callable, List
import numpy as np


def bootstrap_sample(data: np.ndarray) -> np.ndarray:
    """Generate one bootstrap sample (sample with replacement)."""
    # Your solution here
    pass


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    # Your solution here
    pass


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Test if model A is significantly better than model B.

    Returns:
        (difference, p_value)
    """
    # Your solution here
    pass


class BootstrapEvaluator:
    """Evaluation with confidence intervals."""

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        """Initialize evaluator."""
        # Your solution here
        pass

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, Callable]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Evaluate all metrics with CIs.

        Returns:
            {metric_name: (mean, ci_low, ci_high)}
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 15, replace=False)] = 1 - y_pred[np.random.choice(100, 15, replace=False)]

    def accuracy(y_true, y_pred):
        return (y_true == y_pred).mean()

    print("Testing bootstrap confidence intervals...")

    # Test 1: Bootstrap sample
    sample = bootstrap_sample(np.arange(10))
    assert len(sample) == 10, "Test 1 failed"
    print(f"  ✓ Bootstrap sample: {sample}")

    # Test 2: Confidence interval
    mean, ci_low, ci_high = bootstrap_confidence_interval(
        y_true, y_pred, accuracy, n_bootstrap=500
    )
    assert ci_low <= mean <= ci_high, "Test 2 failed"
    print(f"  ✓ Accuracy: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # Test 3: Paired test
    y_pred_worse = y_pred.copy()
    y_pred_worse[np.random.choice(100, 10, replace=False)] = 1
    diff, p_value = paired_bootstrap_test(
        y_true, y_pred, y_pred_worse, accuracy, n_bootstrap=500
    )
    assert 0 <= p_value <= 1, f"Test 3 failed: {p_value}"
    print(f"  ✓ Paired test: diff={diff:.3f}, p={p_value:.3f}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
