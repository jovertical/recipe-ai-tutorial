# Problem 10: Ablation Studies
#
# Implement ablation studies to understand component contributions.
#
# Example:
#   ablation = AblationStudy(full_model, test_data)
#   results = ablation.remove_component("attention")
#   # Returns: {"accuracy": 0.85 -> 0.72, "delta": -0.13}
#
# ML Relevance: Ablation studies:
# - Quantify component importance
# - Justify architecture decisions
# - Identify redundant components
# - Guide model simplification

from typing import List, Dict, Callable, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result of ablating one component."""
    component: str
    baseline_score: float
    ablated_score: float
    delta: float
    relative_change: float


def feature_ablation(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable,
    feature_names: List[str] = None
) -> Dict[str, AblationResult]:
    """
    Ablate each feature and measure impact.

    Args:
        model_fn: Function that trains and returns a model
        X: Feature matrix
        y: Labels
        metric_fn: Evaluation metric
        feature_names: Optional feature names

    Returns:
        {feature_name: AblationResult}
    """
    # Your solution here
    pass


def feature_group_ablation(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable,
    feature_groups: Dict[str, List[int]]
) -> Dict[str, AblationResult]:
    """Ablate groups of related features."""
    # Your solution here
    pass


class AblationStudy:
    """Systematic ablation study framework."""

    def __init__(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable
    ):
        """Initialize ablation study."""
        # Your solution here
        pass

    def baseline_score(self) -> float:
        """Compute baseline with all components."""
        # Your solution here
        pass

    def ablate_feature(self, feature_idx: int) -> AblationResult:
        """Ablate single feature."""
        # Your solution here
        pass

    def ablate_all_features(self) -> List[AblationResult]:
        """Ablate each feature independently."""
        # Your solution here
        pass

    def rank_features_by_importance(self) -> List[Tuple[int, float]]:
        """Rank features by ablation impact."""
        # Your solution here
        pass

    def cumulative_ablation(self, order: List[int]) -> List[float]:
        """
        Remove features one by one in given order.

        Returns:
            Scores after each removal
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    # Simple linear model for testing
    X = np.random.randn(100, 5)
    w = np.array([1.0, 0.5, 0.1, 0.0, -0.5])  # Feature importance varies
    y = (X @ w > 0).astype(int)

    def simple_model_fn(X, y):
        # Just return predictions based on mean
        return lambda X_new: (X_new @ w > 0).astype(int)

    def accuracy(y_true, y_pred):
        return (y_true == y_pred).mean()

    print("Testing ablation studies...")

    # Test 1: Feature ablation
    results = feature_ablation(simple_model_fn, X, y, accuracy)
    assert len(results) == 5, f"Test 1 failed: {len(results)}"
    print(f"  ✓ Ablated {len(results)} features")

    # Test 2: AblationStudy class
    study = AblationStudy(simple_model_fn, X, y, accuracy)
    baseline = study.baseline_score()
    assert 0 <= baseline <= 1, f"Test 2 failed: {baseline}"
    print(f"  ✓ Baseline score: {baseline:.4f}")

    # Test 3: Rank features
    ranking = study.rank_features_by_importance()
    assert len(ranking) == 5, "Test 3 failed"
    print(f"  ✓ Feature ranking: {ranking}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
