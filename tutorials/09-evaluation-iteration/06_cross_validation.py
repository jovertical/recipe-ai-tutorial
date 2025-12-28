# Problem 6: Cross-Validation Strategies
#
# Implement cross-validation for robust model evaluation.
#
# Example:
#   cv = KFoldCV(n_splits=5)
#   scores = cv.evaluate(model, X, y)
#   # Returns: [0.82, 0.85, 0.79, 0.84, 0.81]
#
# ML Relevance: Cross-validation:
# - Provides more reliable performance estimates
# - Detects overfitting
# - Uses all data for both training and validation
# - Essential for small datasets

from typing import List, Dict, Tuple, Iterator, Callable
import numpy as np


def k_fold_split(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold train/test indices.

    Yields:
        (train_indices, test_indices) for each fold
    """
    # Your solution here
    pass


def stratified_k_fold_split(
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold: maintains class proportions in each fold.
    """
    # Your solution here
    pass


class CrossValidator:
    """Cross-validation evaluator."""

    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = False,
        shuffle: bool = True
    ):
        """Initialize cross-validator."""
        # Your solution here
        pass

    def evaluate(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable
    ) -> Dict[str, float]:
        """
        Run cross-validation.

        Args:
            model_fn: Function that returns a trainable model
            X: Features
            y: Labels
            metric_fn: Evaluation metric function

        Returns:
            {"mean": x, "std": y, "scores": [...]}
        """
        # Your solution here
        pass


def leave_one_out_cv(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable
) -> List[float]:
    """Leave-one-out cross-validation."""
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    print("Testing cross-validation...")

    # Test 1: K-fold split
    folds = list(k_fold_split(100, n_splits=5))
    assert len(folds) == 5, f"Test 1a failed: {len(folds)}"
    for train_idx, test_idx in folds:
        assert len(train_idx) == 80, "Test 1b failed"
        assert len(test_idx) == 20, "Test 1c failed"
    print(f"  ✓ K-fold: {len(folds)} folds")

    # Test 2: No overlap between train/test
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0, "Test 2 failed"
    print(f"  ✓ No train/test overlap")

    # Test 3: Stratified split
    y_imbalanced = np.array([0]*80 + [1]*20)
    strat_folds = list(stratified_k_fold_split(y_imbalanced, n_splits=5))
    for train_idx, test_idx in strat_folds:
        test_ratio = y_imbalanced[test_idx].mean()
        assert 0.15 <= test_ratio <= 0.25, f"Test 3 failed: {test_ratio}"
    print(f"  ✓ Stratified maintains class ratio")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
