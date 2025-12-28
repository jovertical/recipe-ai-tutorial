# Problem 5: Ranking Evaluation Metrics
#
# Implement ranking metrics for evaluating recommendation and retrieval systems.
#
# Example:
#   mrr = mean_reciprocal_rank(rankings, relevant_items)
#   ndcg = ndcg_at_k(rankings, relevance_scores, k=10)
#   map_score = mean_average_precision(rankings, relevant_items)
#
# ML Relevance: Ranking metrics evaluate:
# - Search and retrieval quality
# - Recommendation relevance
# - Substitute ordering quality

from typing import List, Dict, Set
import numpy as np


def reciprocal_rank(ranking: List[str], relevant: Set[str]) -> float:
    """
    Compute reciprocal rank: 1/position of first relevant item.

    Args:
        ranking: Ordered list of items
        relevant: Set of relevant items

    Returns:
        Reciprocal rank (0 if no relevant item found)
    """
    # Your solution here
    pass


def mean_reciprocal_rank(
    rankings: List[List[str]],
    relevant_items: List[Set[str]]
) -> float:
    """Compute MRR across multiple queries."""
    # Your solution here
    pass


def precision_at_k(ranking: List[str], relevant: Set[str], k: int) -> float:
    """Precision considering only top-k results."""
    # Your solution here
    pass


def recall_at_k(ranking: List[str], relevant: Set[str], k: int) -> float:
    """Recall considering only top-k results."""
    # Your solution here
    pass


def dcg_at_k(relevance_scores: List[float], k: int = None) -> float:
    """
    Discounted Cumulative Gain.

    DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    """
    # Your solution here
    pass


def ndcg_at_k(
    ranking: List[str],
    relevance: Dict[str, float],
    k: int = None
) -> float:
    """
    Normalized DCG.

    NDCG = DCG / IDCG (ideal DCG)
    """
    # Your solution here
    pass


def mean_average_precision(
    rankings: List[List[str]],
    relevant_items: List[Set[str]]
) -> float:
    """Mean Average Precision across queries."""
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    print("Testing ranking metrics...")

    # Test 1: Reciprocal rank
    ranking = ["a", "b", "c", "d"]
    relevant = {"c", "d"}
    rr = reciprocal_rank(ranking, relevant)
    assert rr == 1/3, f"Test 1 failed: {rr}"
    print(f"  ✓ RR: {rr:.4f}")

    # Test 2: MRR
    rankings = [["a", "b", "c"], ["x", "y", "z"]]
    relevants = [{"b"}, {"z"}]
    mrr = mean_reciprocal_rank(rankings, relevants)
    assert mrr == (0.5 + 1/3) / 2, f"Test 2 failed: {mrr}"
    print(f"  ✓ MRR: {mrr:.4f}")

    # Test 3: Precision@K
    p_at_3 = precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=3)
    assert p_at_3 == 2/3, f"Test 3 failed: {p_at_3}"
    print(f"  ✓ P@3: {p_at_3:.4f}")

    # Test 4: NDCG
    relevance = {"a": 3, "b": 2, "c": 1, "d": 0}
    ndcg = ndcg_at_k(["a", "b", "c", "d"], relevance, k=4)
    assert ndcg == 1.0, f"Test 4 failed: {ndcg}"  # Perfect ranking
    print(f"  ✓ NDCG (perfect): {ndcg:.4f}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
