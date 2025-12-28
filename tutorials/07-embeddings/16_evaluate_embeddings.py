# Problem 16: Evaluate Embeddings
#
# Implement comprehensive evaluation for ingredient embeddings. Good
# embeddings should capture ingredient relationships and be useful for
# downstream tasks like substitution.
#
# You'll implement:
# 1. ingredient_analogy_test() - semantic analogies
# 2. substitution_ranking() - retrieval evaluation with recall@k
# 3. clustering_quality() - category coherence metrics
# 4. EmbeddingEvaluator - comprehensive evaluation suite
#
# Example:
#   analogy_score = ingredient_analogy_test(embeddings)
#   clustering_score = clustering_quality(embeddings, categories)
#   substitution_score = substitution_ranking(embeddings, test_pairs)
#
# Constraints:
#   - Analogy test: does (B - A + C) find D?
#   - Substitution: compute recall@k and MRR
#   - Clustering: intra/inter class similarity, silhouette score
#
# ML Relevance: Embedding evaluation validates training, compares different
# approaches, identifies failure cases, and guides improvements. Use both
# intrinsic (embedding quality) and extrinsic (task performance) metrics.

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def ingredient_analogy_test(embeddings: Dict[str, np.ndarray], analogies: List[Tuple[str, str, str, str]] = None) -> Dict[str, float]:
    """Test analogies: A is to B as C is to D. Returns accuracy@1, accuracy@5."""
    if analogies is None:
        analogies = [
            ("butter", "toast", "jam", "bread"),
            ("olive_oil", "salad", "butter", "bread"),
            ("garlic", "italian", "ginger", "asian"),
            ("beef", "steak", "chicken", "breast"),
            ("milk", "cream", "coconut_milk", "coconut_cream"),
        ]
    # Your solution here
    pass


def substitution_ranking(embeddings: Dict[str, np.ndarray], substitution_pairs: List[Tuple[str, List[str]]], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """Evaluate substitution retrieval: recall@k and MRR."""
    # Your solution here
    pass


def clustering_quality(embeddings: Dict[str, np.ndarray], categories: Dict[str, str]) -> Dict[str, float]:
    """Evaluate clustering: intra/inter class similarity, silhouette, purity."""
    # Your solution here
    pass


def compute_silhouette_score(embeddings: Dict[str, np.ndarray], labels: Dict[str, str]) -> float:
    """Compute silhouette score: (b - a) / max(a, b). Range [-1, 1]."""
    # Your solution here
    pass


def nearest_neighbor_accuracy(embeddings: Dict[str, np.ndarray], categories: Dict[str, str], k: int = 5) -> float:
    """k-NN accuracy: fraction where majority of neighbors have same category."""
    # Your solution here
    pass


def triplet_accuracy(embeddings: Dict[str, np.ndarray], triplets: List[Tuple[str, str, str]]) -> float:
    """Accuracy: fraction where sim(a,p) > sim(a,n)."""
    # Your solution here
    pass


def compute_mrr(embeddings: Dict[str, np.ndarray], queries: List[Tuple[str, str]]) -> float:
    """Mean Reciprocal Rank for (query, expected_target) pairs."""
    # Your solution here
    pass


def embedding_diversity(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Measure diversity: avg_pairwise_similarity, std, isotropy."""
    # Your solution here
    pass


class EmbeddingEvaluator:
    """Comprehensive embedding evaluation suite."""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], categories: Dict[str, str] = None, substitution_pairs: List[Tuple[str, List[str]]] = None):
        """Initialize with embeddings and optional ground truth."""
        # Your solution here
        pass
    
    def evaluate_all(self) -> Dict[str, float]:
        """Run all evaluations."""
        # Your solution here
        pass
    
    def evaluate_intrinsic(self) -> Dict[str, float]:
        """Run intrinsic evaluations (don't need task labels)."""
        # Your solution here
        pass
    
    def evaluate_extrinsic(self) -> Dict[str, float]:
        """Run extrinsic evaluations (need task labels)."""
        # Your solution here
        pass
    
    def compare_with(self, other_embeddings: Dict[str, np.ndarray], name: str = "other") -> Dict[str, Dict[str, float]]:
        """Compare with another set of embeddings."""
        # Your solution here
        pass
    
    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        # Your solution here
        pass


def visualize_evaluation_results(results: Dict[str, float]) -> str:
    """Create text visualization of evaluation results."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    categories = {
        "butter": "fat", "margarine": "fat", "oil": "fat", "lard": "fat",
        "flour": "grain", "rice": "grain", "pasta": "grain",
        "garlic": "aromatic", "onion": "aromatic", "ginger": "aromatic",
        "chicken": "protein", "beef": "protein", "pork": "protein",
        "sugar": "sweetener", "honey": "sweetener"
    }
    
    # Create embeddings where same-category items are similar
    embeddings = {}
    category_vectors = {}
    for cat in set(categories.values()):
        category_vectors[cat] = np.random.randn(64).astype(np.float32)
        category_vectors[cat] /= np.linalg.norm(category_vectors[cat])
    
    for ing, cat in categories.items():
        emb = category_vectors[cat] + np.random.randn(64) * 0.3
        emb = emb / np.linalg.norm(emb)
        embeddings[ing] = emb.astype(np.float32)
    
    substitution_pairs = [
        ("butter", ["margarine", "oil", "lard"]),
        ("flour", ["rice", "pasta"]),
        ("garlic", ["onion", "ginger"]),
        ("chicken", ["beef", "pork"]),
    ]
    
    print("Testing ingredient_analogy_test...")
    
    analogies = [("butter", "margarine", "chicken", "beef")]
    results = ingredient_analogy_test(embeddings, analogies)
    assert isinstance(results, dict), "Test 1a failed"
    assert "accuracy@1" in results or "accuracy" in results, "Test 1b failed"
    print(f"  ✓ Analogy test completed")
    
    print("\nTesting substitution_ranking...")
    
    results = substitution_ranking(embeddings, substitution_pairs, k_values=[1, 3, 5])
    assert "recall@1" in results, "Test 2a failed"
    assert "recall@5" in results, "Test 2b failed"
    assert "mrr" in results, "Test 2c failed"
    assert 0 <= results["recall@5"] <= 1, f"Test 2d failed: {results['recall@5']}"
    print(f"  ✓ Substitution ranking: recall@5={results['recall@5']:.2%}")
    
    print("\nTesting clustering_quality...")
    
    results = clustering_quality(embeddings, categories)
    assert "intra_class_similarity" in results, "Test 3a failed"
    assert "inter_class_similarity" in results, "Test 3b failed"
    print(f"  ✓ Intra: {results['intra_class_similarity']:.3f}, Inter: {results['inter_class_similarity']:.3f}")
    
    print("\nTesting compute_silhouette_score...")
    
    score = compute_silhouette_score(embeddings, categories)
    assert -1 <= score <= 1, f"Test 4 failed: {score}"
    print(f"  ✓ Silhouette score: {score:.3f}")
    
    print("\nTesting nearest_neighbor_accuracy...")
    
    acc = nearest_neighbor_accuracy(embeddings, categories, k=3)
    assert 0 <= acc <= 1, f"Test 5 failed: {acc}"
    print(f"  ✓ 3-NN accuracy: {acc:.2%}")
    
    print("\nTesting triplet_accuracy...")
    
    triplets = [
        ("butter", "margarine", "garlic"),
        ("garlic", "onion", "butter"),
        ("chicken", "beef", "flour"),
    ]
    acc = triplet_accuracy(embeddings, triplets)
    assert 0 <= acc <= 1, f"Test 6 failed: {acc}"
    print(f"  ✓ Triplet accuracy: {acc:.2%}")
    
    print("\nTesting compute_mrr...")
    
    queries = [("butter", "margarine"), ("garlic", "onion")]
    mrr = compute_mrr(embeddings, queries)
    assert 0 <= mrr <= 1, f"Test 7 failed: {mrr}"
    print(f"  ✓ MRR: {mrr:.3f}")
    
    print("\nTesting embedding_diversity...")
    
    diversity = embedding_diversity(embeddings)
    assert "avg_pairwise_similarity" in diversity, "Test 8a failed"
    assert "isotropy" in diversity, "Test 8b failed"
    print(f"  ✓ Avg pairwise similarity: {diversity['avg_pairwise_similarity']:.3f}")
    
    print("\nTesting EmbeddingEvaluator...")
    
    evaluator = EmbeddingEvaluator(embeddings, categories, substitution_pairs)
    print("  ✓ Evaluator initialized")
    
    all_results = evaluator.evaluate_all()
    assert len(all_results) > 0, "Test 10 failed"
    print(f"  ✓ Evaluated {len(all_results)} metrics")
    
    intrinsic = evaluator.evaluate_intrinsic()
    assert len(intrinsic) > 0, "Test 11 failed"
    print(f"  ✓ Intrinsic: {len(intrinsic)} metrics")
    
    extrinsic = evaluator.evaluate_extrinsic()
    assert len(extrinsic) > 0, "Test 12 failed"
    print(f"  ✓ Extrinsic: {len(extrinsic)} metrics")
    
    random_embeddings = {
        ing: np.random.randn(64).astype(np.float32)
        for ing in embeddings
    }
    for ing in random_embeddings:
        random_embeddings[ing] /= np.linalg.norm(random_embeddings[ing])
    
    comparison = evaluator.compare_with(random_embeddings, "random")
    assert "current" in comparison, "Test 13a failed"
    assert "random" in comparison, "Test 13b failed"
    print("  ✓ Comparison works")
    
    report = evaluator.generate_report()
    assert isinstance(report, str), "Test 14a failed"
    assert len(report) > 0, "Test 14b failed"
    print("  ✓ Report generated")
    
    print("\nTesting visualize_evaluation_results...")
    
    viz = visualize_evaluation_results(all_results)
    assert isinstance(viz, str), "Test 15 failed"
    print("  ✓ Visualization created")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\n--- Evaluation Summary ---")
    print(f"Triplet Accuracy: {triplet_accuracy(embeddings, triplets):.2%}")
    print(f"3-NN Accuracy: {nearest_neighbor_accuracy(embeddings, categories, k=3):.2%}")
    print(f"Silhouette Score: {compute_silhouette_score(embeddings, categories):.3f}")
    print(f"Substitution Recall@5: {substitution_ranking(embeddings, substitution_pairs)['recall@5']:.2%}")
