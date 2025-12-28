# Problem 11: Statistical Model Comparison
#
# Implement statistical tests to compare model performance.
#
# Example:
#   comparator = ModelComparator()
#   result = comparator.mcnemar_test(model_a_preds, model_b_preds, y_true)
#   # Returns: {"chi2": 5.2, "p_value": 0.022, "significant": True}
#
# ML Relevance: Statistical comparison:
# - Determines if improvements are real
# - Accounts for variance
# - Prevents overfitting to test set
# - Supports publication claims

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    statistic: float
    p_value: float
    significant: bool
    winner: str = None
    effect_size: float = None


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    McNemar's test for comparing two classifiers.

    Tests if the disagreements are symmetric.
    """
    # Your solution here
    pass


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Paired t-test for comparing model scores across folds.
    """
    # Your solution here
    pass


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Non-parametric alternative to paired t-test.
    """
    # Your solution here
    pass


class ModelComparator:
    """Compare multiple models statistically."""

    def __init__(self, alpha: float = 0.05):
        """Initialize comparator."""
        self.alpha = alpha

    def pairwise_comparison(
        self,
        model_scores: Dict[str, np.ndarray],
        test_type: str = "paired_t"
    ) -> Dict[Tuple[str, str], ComparisonResult]:
        """
        Compare all pairs of models.

        Returns:
            {(model_a, model_b): ComparisonResult}
        """
        # Your solution here
        pass

    def rank_models(
        self,
        model_scores: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float, int]]:
        """
        Rank models by mean score and wins.

        Returns:
            [(model_name, mean_score, num_significant_wins)]
        """
        # Your solution here
        pass

    def bonferroni_correction(
        self,
        p_values: List[float]
    ) -> List[bool]:
        """Apply Bonferroni correction for multiple comparisons."""
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    # Model A slightly better than B
    y_true = np.random.randint(0, 2, 100)
    pred_a = y_true.copy()
    pred_a[np.random.choice(100, 10, replace=False)] ^= 1
    pred_b = y_true.copy()
    pred_b[np.random.choice(100, 20, replace=False)] ^= 1

    print("Testing model comparison...")

    # Test 1: McNemar test
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert 0 <= result.p_value <= 1, f"Test 1 failed: {result.p_value}"
    print(f"  ✓ McNemar: chi2={result.statistic:.2f}, p={result.p_value:.4f}")

    # Test 2: Paired t-test (on CV scores)
    scores_a = np.array([0.85, 0.87, 0.82, 0.88, 0.84])
    scores_b = np.array([0.80, 0.82, 0.79, 0.81, 0.78])
    result = paired_t_test(scores_a, scores_b)
    assert result.p_value < 0.05, "Test 2 failed: should be significant"
    print(f"  ✓ Paired t-test: t={result.statistic:.2f}, p={result.p_value:.4f}")

    # Test 3: ModelComparator
    comparator = ModelComparator()
    model_scores = {"A": scores_a, "B": scores_b}
    ranking = comparator.rank_models(model_scores)
    assert ranking[0][0] == "A", "Test 3 failed: A should rank first"
    print(f"  ✓ Ranking: {ranking}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
