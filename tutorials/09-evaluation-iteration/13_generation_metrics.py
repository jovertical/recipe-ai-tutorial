# Problem 13: Text Generation Metrics
#
# Implement metrics for evaluating generated recipe text.
#
# Example:
#   bleu = compute_bleu(generated_text, reference_texts)
#   rouge = compute_rouge(generated_text, reference_text)
#
# ML Relevance: Generation metrics:
# - Evaluate recipe instruction quality
# - Compare different generation approaches
# - Track improvement over training

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.lower().split()


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    # Your solution here
    pass


def bleu_score(
    candidate: str,
    references: List[str],
    max_n: int = 4,
    weights: List[float] = None
) -> float:
    """
    Compute BLEU score for generated text.

    Args:
        candidate: Generated text
        references: Reference texts
        max_n: Maximum n-gram size
        weights: Weights for each n-gram level

    Returns:
        BLEU score (0 to 1)
    """
    # Your solution here
    pass


def rouge_n(candidate: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N score.

    Returns:
        {"precision": x, "recall": y, "f1": z}
    """
    # Your solution here
    pass


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L (longest common subsequence).
    """
    # Your solution here
    pass


def distinct_n(text: str, n: int = 2) -> float:
    """
    Compute distinct-n: ratio of unique n-grams.

    Measures generation diversity.
    """
    # Your solution here
    pass


class GenerationEvaluator:
    """Evaluate generated recipe text."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate(
        self,
        generated: str,
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all generation metrics.

        Returns:
            {"bleu": x, "rouge1_f1": y, "rouge_l_f1": z, "distinct2": w}
        """
        # Your solution here
        pass

    def batch_evaluate(
        self,
        generated_list: List[str],
        references_list: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate batch and return averages."""
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    print("Testing generation metrics...")

    candidate = "mix flour sugar and butter"
    reference = "mix flour sugar and eggs"
    references = [reference, "combine flour with sugar and butter"]

    # Test 1: N-grams
    tokens = tokenize(candidate)
    bigrams = ngrams(tokens, 2)
    assert len(bigrams) == 4, f"Test 1 failed: {len(bigrams)}"
    print(f"  ✓ Bigrams: {bigrams}")

    # Test 2: BLEU
    bleu = bleu_score(candidate, references)
    assert 0 <= bleu <= 1, f"Test 2 failed: {bleu}"
    print(f"  ✓ BLEU: {bleu:.4f}")

    # Test 3: ROUGE-1
    rouge = rouge_n(candidate, reference, n=1)
    assert "f1" in rouge, "Test 3 failed"
    print(f"  ✓ ROUGE-1: {rouge}")

    # Test 4: ROUGE-L
    rouge_l_score = rouge_l(candidate, reference)
    assert "f1" in rouge_l_score, "Test 4 failed"
    print(f"  ✓ ROUGE-L: {rouge_l_score}")

    # Test 5: Distinct-2
    diverse_text = "one two three four five six seven eight"
    d2 = distinct_n(diverse_text, n=2)
    assert d2 == 1.0, f"Test 5 failed: {d2}"  # All unique bigrams
    print(f"  ✓ Distinct-2: {d2:.4f}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
