# Problem 12: Recipe-Specific Evaluation Metrics
#
# Implement domain-specific metrics for Recipe AI evaluation.
#
# Example:
#   metrics = RecipeMetrics()
#   score = metrics.ingredient_overlap(predicted_recipe, reference_recipe)
#   coherence = metrics.recipe_coherence(predicted_recipe)
#
# ML Relevance: Domain-specific metrics:
# - Capture what matters for the application
# - Generic metrics miss domain nuances
# - Enable meaningful comparison

from typing import List, Dict, Set, Tuple
import numpy as np


def ingredient_precision(predicted: Set[str], reference: Set[str]) -> float:
    """What fraction of predicted ingredients are correct?"""
    # Your solution here
    pass


def ingredient_recall(predicted: Set[str], reference: Set[str]) -> float:
    """What fraction of reference ingredients are predicted?"""
    # Your solution here
    pass


def ingredient_f1(predicted: Set[str], reference: Set[str]) -> float:
    """F1 score for ingredient prediction."""
    # Your solution here
    pass


def substitution_accuracy(
    predictions: List[Tuple[str, str]],
    ground_truth: Dict[str, List[str]]
) -> float:
    """
    Accuracy for substitution predictions.

    Args:
        predictions: List of (original, predicted_substitute)
        ground_truth: {original: [valid_substitutes]}
    """
    # Your solution here
    pass


class RecipeMetrics:
    """Domain-specific recipe evaluation."""

    def __init__(self, ingredient_vocab: Set[str] = None):
        """Initialize with optional ingredient vocabulary."""
        # Your solution here
        pass

    def recipe_coherence(self, ingredients: List[str]) -> float:
        """
        Score how coherent a set of ingredients is.

        High score = ingredients commonly appear together.
        """
        # Your solution here
        pass

    def dietary_compliance(
        self,
        ingredients: List[str],
        dietary_restriction: str
    ) -> float:
        """Check if recipe complies with dietary restriction."""
        # Your solution here
        pass

    def nutritional_balance(self, ingredients: List[str]) -> Dict[str, float]:
        """Assess nutritional balance of recipe."""
        # Your solution here
        pass

    def evaluate_substitution(
        self,
        original: str,
        substitute: str,
        context: List[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive substitution evaluation.

        Returns:
            {"flavor_match": x, "texture_match": y, "function_match": z}
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    print("Testing recipe-specific metrics...")

    # Test 1: Ingredient precision/recall
    pred = {"flour", "sugar", "butter", "salt"}
    ref = {"flour", "sugar", "butter", "eggs"}

    prec = ingredient_precision(pred, ref)
    assert prec == 0.75, f"Test 1a failed: {prec}"
    rec = ingredient_recall(pred, ref)
    assert rec == 0.75, f"Test 1b failed: {rec}"
    print(f"  ✓ Precision: {prec:.2f}, Recall: {rec:.2f}")

    # Test 2: Substitution accuracy
    predictions = [("butter", "margarine"), ("egg", "flax")]
    ground_truth = {"butter": ["margarine", "coconut_oil"], "egg": ["flax", "chia"]}
    acc = substitution_accuracy(predictions, ground_truth)
    assert acc == 1.0, f"Test 2 failed: {acc}"
    print(f"  ✓ Substitution accuracy: {acc:.2f}")

    # Test 3: RecipeMetrics
    metrics = RecipeMetrics()
    coherence = metrics.recipe_coherence(["flour", "sugar", "butter", "eggs"])
    assert 0 <= coherence <= 1, f"Test 3 failed: {coherence}"
    print(f"  ✓ Recipe coherence: {coherence:.2f}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
