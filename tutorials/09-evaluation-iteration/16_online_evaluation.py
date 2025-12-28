# Problem 16: Online Evaluation and A/B Testing
#
# Implement online evaluation for production model assessment.
#
# Example:
#   ab_test = ABTest(control_model, treatment_model)
#   ab_test.assign_user(user_id)  # Randomly assigns to group
#   result = ab_test.analyze(metrics_data)
#   # Returns: {"lift": 0.05, "p_value": 0.03, "significant": True}
#
# ML Relevance: Online evaluation:
# - Tests real user behavior
# - Catches issues offline metrics miss
# - Required for production decisions

from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass
import hashlib


@dataclass
class ExperimentConfig:
    """A/B test configuration."""
    name: str
    control_ratio: float = 0.5
    min_sample_size: int = 1000
    significance_level: float = 0.05


def assign_variant(
    user_id: str,
    experiment_name: str,
    control_ratio: float = 0.5
) -> str:
    """
    Deterministically assign user to variant.

    Same user always gets same variant (for consistency).
    """
    # Your solution here
    pass


def compute_lift(
    control_metric: float,
    treatment_metric: float
) -> float:
    """Compute relative lift: (treatment - control) / control."""
    # Your solution here
    pass


def z_test_proportions(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int
) -> Dict[str, float]:
    """
    Z-test for comparing proportions (e.g., conversion rates).

    Returns:
        {"z_stat": x, "p_value": y}
    """
    # Your solution here
    pass


def t_test_means(
    values_a: np.ndarray,
    values_b: np.ndarray
) -> Dict[str, float]:
    """
    T-test for comparing means (e.g., average rating).

    Returns:
        {"t_stat": x, "p_value": y}
    """
    # Your solution here
    pass


class ABTest:
    """A/B test framework."""

    def __init__(self, config: ExperimentConfig):
        """Initialize A/B test."""
        # Your solution here
        pass

    def assign(self, user_id: str) -> str:
        """Assign user to control or treatment."""
        # Your solution here
        pass

    def log_event(
        self,
        user_id: str,
        event_type: str,
        value: float = 1.0
    ):
        """Log an event for analysis."""
        # Your solution here
        pass

    def analyze(self, metric: str = "conversion") -> Dict[str, any]:
        """
        Analyze experiment results.

        Returns:
            {
                "control_rate": x,
                "treatment_rate": y,
                "lift": z,
                "p_value": p,
                "significant": bool,
                "sample_sizes": {"control": n1, "treatment": n2}
            }
        """
        # Your solution here
        pass

    def is_ready(self) -> bool:
        """Check if enough data collected for analysis."""
        # Your solution here
        pass

    def stopping_decision(self) -> str:
        """
        Recommend whether to stop experiment.

        Returns: "continue", "stop_winner", or "stop_no_effect"
        """
        # Your solution here
        pass


class MultiArmedBandit:
    """Thompson Sampling for adaptive experiments."""

    def __init__(self, n_variants: int):
        """Initialize bandit with n variants."""
        # Your solution here
        pass

    def select(self) -> int:
        """Select variant using Thompson Sampling."""
        # Your solution here
        pass

    def update(self, variant: int, reward: float):
        """Update beliefs based on observed reward."""
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    print("Testing online evaluation...")

    # Test 1: Deterministic assignment
    v1 = assign_variant("user123", "test_exp")
    v2 = assign_variant("user123", "test_exp")
    assert v1 == v2, "Test 1 failed: assignment not deterministic"
    print(f"  ✓ User assignment deterministic: {v1}")

    # Test 2: Lift computation
    lift = compute_lift(control_metric=0.10, treatment_metric=0.12)
    assert abs(lift - 0.20) < 0.01, f"Test 2 failed: {lift}"
    print(f"  ✓ Lift: {lift:.1%}")

    # Test 3: Z-test
    result = z_test_proportions(100, 1000, 120, 1000)
    assert "p_value" in result, "Test 3 failed"
    print(f"  ✓ Z-test: z={result['z_stat']:.2f}, p={result['p_value']:.4f}")

    # Test 4: ABTest
    config = ExperimentConfig(name="recipe_model_v2", min_sample_size=100)
    ab = ABTest(config)
    assignments = [ab.assign(f"user_{i}") for i in range(200)]
    control_count = sum(1 for a in assignments if a == "control")
    assert 80 < control_count < 120, f"Test 4 failed: {control_count}"
    print(f"  ✓ AB split: {control_count} control, {200-control_count} treatment")

    # Test 5: Multi-armed bandit
    bandit = MultiArmedBandit(n_variants=3)
    for _ in range(100):
        v = bandit.select()
        reward = 1.0 if v == 0 else 0.5  # Variant 0 is best
        bandit.update(v, reward)
    # After learning, should prefer variant 0
    selections = [bandit.select() for _ in range(50)]
    assert selections.count(0) > 20, "Test 5 failed: should prefer best variant"
    print(f"  ✓ Bandit learned: variant 0 selected {selections.count(0)}/50 times")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
