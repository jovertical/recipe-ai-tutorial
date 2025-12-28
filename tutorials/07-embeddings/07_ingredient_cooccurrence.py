# Problem 7: Ingredient Co-occurrence
#
# Build ingredient embeddings using co-occurrence statistics. This is a
# classical approach that works well for structured data like recipes.
#
# You'll implement:
# 1. build_cooccurrence_matrix() - count ingredient pairs
# 2. pmi_matrix() - pointwise mutual information
# 3. svd_embeddings() - matrix factorization embeddings
# 4. Analysis and substitution functions
#
# Example:
#   Recipes: ["flour, sugar, butter", "flour, yeast, salt", "butter, garlic"]
#   Co-occurrence matrix shows flour+butter appear together often.
#   PMI reveals which pairs appear more than expected by chance.
#   SVD creates dense embeddings from the sparse matrix.
#
# Constraints:
#   - Matrix must be symmetric (co-occurrence is bidirectional)
#   - Use PPMI (positive PMI) to avoid negative values
#   - SVD embeddings use U * sqrt(S) for balanced representation
#
# ML Relevance: Co-occurrence captures which ingredients appear together.
# PMI measures association strength beyond raw counts. SVD creates dense
# embeddings from sparse co-occurrence, similar to GloVe and LSA.

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import Counter


def build_cooccurrence_matrix(recipes: List[List[str]], vocabulary: Dict[str, int] = None, min_count: int = 1) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build symmetric co-occurrence matrix from recipes."""
    # Your solution here
    pass


def compute_ingredient_frequencies(recipes: List[List[str]], vocabulary: Dict[str, int]) -> np.ndarray:
    """Count how often each ingredient appears."""
    # Your solution here
    pass


def pmi_matrix(cooccurrence: np.ndarray, frequencies: np.ndarray, total_recipes: int, positive: bool = True) -> np.ndarray:
    """Compute PMI: log(P(x,y) / (P(x) * P(y))). Use PPMI if positive=True."""
    # Your solution here
    pass


def svd_embeddings(matrix: np.ndarray, embedding_dim: int = 50) -> np.ndarray:
    """Create embeddings using SVD: U * sqrt(S)."""
    # Your solution here
    pass


def find_related_ingredients(ingredient: str, embeddings: np.ndarray, vocabulary: Dict[str, int], top_k: int = 10) -> List[Tuple[str, float]]:
    """Find ingredients similar to the query."""
    # Your solution here
    pass


def ingredient_pair_score(ing1: str, ing2: str, pmi: np.ndarray, vocabulary: Dict[str, int]) -> float:
    """Get PMI score for an ingredient pair."""
    # Your solution here
    pass


def find_substitute_candidates(ingredient: str, embeddings: np.ndarray, vocabulary: Dict[str, int], category: str = None, categories: Dict[str, str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
    """Find potential substitutes, optionally filtering by category."""
    # Your solution here
    pass


def analyze_cooccurrence_statistics(cooccurrence: np.ndarray, vocabulary: Dict[str, int]) -> Dict[str, any]:
    """Analyze sparsity, most connected ingredients, common pairs."""
    # Your solution here
    pass


def create_ingredient_graph(cooccurrence: np.ndarray, vocabulary: Dict[str, int], threshold: int = 5) -> Dict[str, List[str]]:
    """Create adjacency list graph from co-occurrence."""
    # Your solution here
    pass


def smooth_cooccurrence(cooccurrence: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """Apply power smoothing to reduce frequent pair dominance."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    recipes = [
        ["flour", "sugar", "butter", "eggs"],
        ["flour", "yeast", "salt", "water"],
        ["butter", "garlic", "herbs", "olive_oil"],
        ["flour", "butter", "sugar", "vanilla"],
        ["garlic", "olive_oil", "tomato", "basil"],
        ["flour", "eggs", "milk", "butter"],
        ["chicken", "garlic", "herbs", "olive_oil"],
        ["pasta", "tomato", "basil", "garlic"]
    ]
    
    print("Testing build_cooccurrence_matrix...")
    
    # Test 1: Build matrix
    cooc, vocab = build_cooccurrence_matrix(recipes)
    assert cooc.shape[0] == cooc.shape[1], "Test 1a failed: should be square"
    assert cooc.shape[0] == len(vocab), f"Test 1b failed: {cooc.shape} vs {len(vocab)}"
    print("  ✓ Matrix built")
    
    # Test 2: Check co-occurrences
    flour_idx = vocab["flour"]
    butter_idx = vocab["butter"]
    assert cooc[flour_idx, butter_idx] == 3, f"Test 2 failed: {cooc[flour_idx, butter_idx]}"
    print("  ✓ Co-occurrence counts correct")
    
    # Test 3: Symmetry
    assert cooc[flour_idx, butter_idx] == cooc[butter_idx, flour_idx], "Test 3 failed"
    print("  ✓ Matrix is symmetric")
    
    print("\nTesting compute_ingredient_frequencies...")
    
    # Test 4: Frequencies
    freqs = compute_ingredient_frequencies(recipes, vocab)
    assert len(freqs) == len(vocab), "Test 4a failed"
    assert freqs[vocab["flour"]] == 4, f"Test 4b failed: {freqs[vocab['flour']]}"
    assert freqs[vocab["garlic"]] == 4, f"Test 4c failed: {freqs[vocab['garlic']]}"
    print("  ✓ Frequencies computed")
    
    print("\nTesting pmi_matrix...")
    
    # Test 5: PMI matrix
    pmi = pmi_matrix(cooc, freqs, len(recipes), positive=True)
    assert pmi.shape == cooc.shape, "Test 5a failed"
    assert np.all(pmi >= 0), "Test 5b failed: PPMI should be non-negative"
    print("  ✓ PMI matrix computed")
    
    # Test 6: PMI values make sense
    flour_sugar_pmi = pmi[vocab["flour"], vocab["sugar"]]
    assert flour_sugar_pmi > 0, f"Test 6 failed: {flour_sugar_pmi}"
    print("  ✓ PMI values reasonable")
    
    print("\nTesting svd_embeddings...")
    
    # Test 7: SVD embeddings
    embeddings = svd_embeddings(pmi, embedding_dim=5)
    assert embeddings.shape == (len(vocab), 5), f"Test 7 failed: {embeddings.shape}"
    print("  ✓ SVD embeddings created")
    
    print("\nTesting find_related_ingredients...")
    
    # Test 8: Find related
    related = find_related_ingredients("flour", embeddings, vocab, top_k=3)
    assert len(related) == 3, f"Test 8a failed: {len(related)}"
    assert all(isinstance(r, tuple) and len(r) == 2 for r in related), "Test 8b failed"
    print("  ✓ Related ingredients found")
    
    print("\nTesting ingredient_pair_score...")
    
    # Test 9: Pair score
    score = ingredient_pair_score("flour", "sugar", pmi, vocab)
    assert isinstance(score, (int, float)), f"Test 9 failed: {type(score)}"
    print("  ✓ Pair score computed")
    
    print("\nTesting find_substitute_candidates...")
    
    # Test 10: Find substitutes
    subs = find_substitute_candidates("butter", embeddings, vocab, top_k=3)
    assert len(subs) == 3, f"Test 10a failed: {len(subs)}"
    assert "butter" not in [s for s, _ in subs], "Test 10b: should exclude self"
    print("  ✓ Substitutes found")
    
    print("\nTesting analyze_cooccurrence_statistics...")
    
    # Test 11: Statistics
    stats = analyze_cooccurrence_statistics(cooc, vocab)
    assert "sparsity" in stats, "Test 11a failed"
    assert "most_connected" in stats, "Test 11b failed"
    assert 0 <= stats["sparsity"] <= 1, f"Test 11c failed: {stats['sparsity']}"
    print("  ✓ Statistics computed")
    
    print("\nTesting create_ingredient_graph...")
    
    # Test 12: Graph creation
    graph = create_ingredient_graph(cooc, vocab, threshold=2)
    assert isinstance(graph, dict), "Test 12a failed"
    if "flour" in graph:
        assert "butter" in graph["flour"], f"Test 12b failed: {graph['flour']}"
    print("  ✓ Graph created")
    
    print("\nTesting smooth_cooccurrence...")
    
    # Test 13: Smoothing
    smoothed = smooth_cooccurrence(cooc, alpha=0.75)
    assert smoothed.shape == cooc.shape, "Test 13a failed"
    print("  ✓ Smoothing applied")
    
    # Test 14: Full pipeline
    print("\nTesting full pipeline...")
    cooc, vocab = build_cooccurrence_matrix(recipes, min_count=1)
    freqs = compute_ingredient_frequencies(recipes, vocab)
    pmi = pmi_matrix(cooc, freqs, len(recipes), positive=True)
    embeddings = svd_embeddings(pmi, embedding_dim=10)
    
    similar = find_related_ingredients("butter", embeddings, vocab, top_k=5)
    print(f"  Ingredients similar to 'butter': {[s[0] for s in similar[:3]]}")
    
    similar = find_related_ingredients("garlic", embeddings, vocab, top_k=5)
    print(f"  Ingredients similar to 'garlic': {[s[0] for s in similar[:3]]}")
    print("  ✓ Full pipeline works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
