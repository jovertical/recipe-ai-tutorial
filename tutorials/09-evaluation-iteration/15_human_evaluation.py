# Problem 15: Human Evaluation Framework
#
# Design and analyze human evaluations for recipe quality.
#
# Example:
#   study = HumanEvaluation(samples, evaluators=5)
#   study.compute_agreement()
#   # Returns: {"fleiss_kappa": 0.72, "interpretation": "substantial"}
#
# ML Relevance: Human evaluation:
# - Ground truth for subjective tasks
# - Validates automated metrics
# - Essential for deployment decisions

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Annotation:
    """Single human annotation."""
    sample_id: str
    annotator_id: str
    rating: int  # e.g., 1-5
    labels: Dict[str, any] = None


def cohens_kappa(
    ratings_a: np.ndarray,
    ratings_b: np.ndarray
) -> float:
    """
    Cohen's Kappa for inter-annotator agreement (2 raters).

    kappa = (po - pe) / (1 - pe)
    where po = observed agreement, pe = expected by chance
    """
    # Your solution here
    pass


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Fleiss' Kappa for multiple raters.

    Args:
        ratings_matrix: (n_samples, n_categories) count of raters per category
    """
    # Your solution here
    pass


def krippendorff_alpha(
    ratings: Dict[str, Dict[str, int]]
) -> float:
    """
    Krippendorff's Alpha for reliability.

    Args:
        ratings: {annotator_id: {sample_id: rating}}
    """
    # Your solution here
    pass


class HumanEvaluation:
    """Human evaluation study framework."""

    def __init__(self, annotations: List[Annotation]):
        """Initialize with annotations."""
        # Your solution here
        pass

    def inter_annotator_agreement(self) -> Dict[str, float]:
        """
        Compute agreement metrics.

        Returns:
            {"fleiss_kappa": x, "mean_pairwise_kappa": y}
        """
        # Your solution here
        pass

    def annotator_reliability(self) -> Dict[str, float]:
        """
        Score each annotator's reliability.

        Returns:
            {annotator_id: reliability_score}
        """
        # Your solution here
        pass

    def aggregate_ratings(
        self,
        method: str = "majority"
    ) -> Dict[str, int]:
        """
        Aggregate multiple ratings per sample.

        Methods: "majority", "mean", "median", "dawid_skene"
        """
        # Your solution here
        pass

    def required_sample_size(
        self,
        effect_size: float = 0.3,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """Estimate required samples for statistical power."""
        # Your solution here
        pass


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value."""
    if kappa < 0:
        return "poor"
    elif kappa < 0.2:
        return "slight"
    elif kappa < 0.4:
        return "fair"
    elif kappa < 0.6:
        return "moderate"
    elif kappa < 0.8:
        return "substantial"
    else:
        return "almost_perfect"


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    print("Testing human evaluation framework...")

    # Test 1: Cohen's Kappa
    # Perfect agreement
    ratings_a = np.array([1, 2, 3, 1, 2])
    ratings_b = np.array([1, 2, 3, 1, 2])
    kappa = cohens_kappa(ratings_a, ratings_b)
    assert kappa == 1.0, f"Test 1a failed: {kappa}"

    # Partial agreement
    ratings_b = np.array([1, 2, 3, 2, 2])
    kappa = cohens_kappa(ratings_a, ratings_b)
    assert 0 < kappa < 1, f"Test 1b failed: {kappa}"
    print(f"  ✓ Cohen's Kappa: {kappa:.4f}")

    # Test 2: Fleiss' Kappa
    # 5 samples, 3 categories, ratings from 4 raters
    ratings_matrix = np.array([
        [4, 0, 0],  # All agree on cat 0
        [3, 1, 0],  # Mostly cat 0
        [0, 4, 0],  # All agree on cat 1
        [1, 2, 1],  # Mixed
        [0, 0, 4],  # All agree on cat 2
    ])
    fkappa = fleiss_kappa(ratings_matrix)
    assert 0 < fkappa <= 1, f"Test 2 failed: {fkappa}"
    print(f"  ✓ Fleiss' Kappa: {fkappa:.4f} ({interpret_kappa(fkappa)})")

    # Test 3: HumanEvaluation
    annotations = [
        Annotation("s1", "a1", 4), Annotation("s1", "a2", 4), Annotation("s1", "a3", 5),
        Annotation("s2", "a1", 2), Annotation("s2", "a2", 3), Annotation("s2", "a3", 2),
    ]
    study = HumanEvaluation(annotations)
    agreement = study.inter_annotator_agreement()
    assert "fleiss_kappa" in agreement, "Test 3 failed"
    print(f"  ✓ Agreement: {agreement}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
