# Problem 8: Test Set Design
#
# Design test sets that accurately measure real-world performance.
#
# Example:
#   splitter = TestSetDesigner(data)
#   train, test = splitter.temporal_split(test_ratio=0.2)
#   test_hard = splitter.get_hard_cases(model)
#
# ML Relevance: Good test sets:
# - Reflect production distribution
# - Include edge cases and hard examples
# - Avoid data leakage
# - Enable fair model comparison

from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Train/test split with metadata."""
    train_indices: np.ndarray
    test_indices: np.ndarray
    split_method: str
    metadata: Dict = None


def temporal_split(
    timestamps: np.ndarray,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split by time: older data for training, newer for testing.

    Prevents future data leakage.
    """
    # Your solution here
    pass


def stratified_split(
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Split maintaining class distribution."""
    # Your solution here
    pass


def group_split(
    groups: np.ndarray,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split by groups: all items in a group go to same set.

    E.g., all recipes from same user in train OR test, not both.
    """
    # Your solution here
    pass


class TestSetDesigner:
    """Design comprehensive test sets."""

    def __init__(self, n_samples: int):
        """Initialize designer."""
        # Your solution here
        pass

    def random_split(self, test_ratio: float = 0.2) -> DataSplit:
        """Simple random split."""
        # Your solution here
        pass

    def create_hard_cases_split(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        hard_ratio: float = 0.3
    ) -> np.ndarray:
        """
        Identify hard cases (misclassified or low confidence).

        Returns:
            Indices of hard cases
        """
        # Your solution here
        pass

    def create_slice_based_test(
        self,
        features: Dict[str, np.ndarray],
        slices: Dict[str, callable]
    ) -> Dict[str, np.ndarray]:
        """
        Create test subsets for different data slices.

        Example slices: short_text, long_text, rare_class, etc.
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 100
    timestamps = np.arange(n_samples)
    y = np.random.randint(0, 3, n_samples)
    groups = np.repeat(np.arange(20), 5)

    print("Testing test set design...")

    # Test 1: Temporal split
    train_idx, test_idx = temporal_split(timestamps, test_ratio=0.2)
    assert max(train_idx) < min(test_idx), "Test 1 failed: temporal order violated"
    print(f"  ✓ Temporal split: train max={max(train_idx)}, test min={min(test_idx)}")

    # Test 2: Stratified split
    train_idx, test_idx = stratified_split(y, test_ratio=0.2)
    train_dist = np.bincount(y[train_idx]) / len(train_idx)
    test_dist = np.bincount(y[test_idx]) / len(test_idx)
    assert np.allclose(train_dist, test_dist, atol=0.1), "Test 2 failed"
    print(f"  ✓ Stratified split maintains distribution")

    # Test 3: Group split
    train_idx, test_idx = group_split(groups, test_ratio=0.2)
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert len(train_groups & test_groups) == 0, "Test 3 failed: group leakage"
    print(f"  ✓ Group split: no group overlap")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
