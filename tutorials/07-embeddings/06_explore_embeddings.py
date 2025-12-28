# Problem 6: Explore Embeddings
#
# Build tools to explore and understand what your trained embeddings have
# learned. Visualization and analysis help validate embedding quality.
#
# You'll implement:
# 1. find_nearest_neighbors() - find similar words
# 2. analogy() - word vector arithmetic
# 3. visualize_embeddings() - 2D visualization with PCA
# 4. cluster_words() - group similar words
#
# Example:
#   find_nearest("butter") → ["margarine", "oil", "shortening", "lard"]
#   analogy("flour", "bread", "rice") → ["sushi", "risotto"]
#   visualize_clusters(["beef", "chicken", "carrot", "onion"])
#   → Shows proteins cluster separately from vegetables
#
# Constraints:
#   - Use cosine similarity for nearest neighbors
#   - Implement simple k-means for clustering
#   - Use PCA for dimensionality reduction
#
# ML Relevance: Embedding exploration helps you validate training quality,
# understand what the model learned, debug issues (unrelated words clustering),
# and communicate results to stakeholders.

from typing import List, Dict, Tuple, Set
import numpy as np


def find_nearest_neighbors(word: str, embeddings: Dict[str, np.ndarray], top_k: int = 10, exclude_self: bool = True) -> List[Tuple[str, float]]:
    """Find words most similar to the query word."""
    # Your solution here
    pass


def analogy(word_a: str, word_b: str, word_c: str, embeddings: Dict[str, np.ndarray], top_k: int = 5, exclude_input: bool = True) -> List[Tuple[str, float]]:
    """Solve analogy: A is to B as C is to ? (using B - A + C)."""
    # Your solution here
    pass


def find_outlier(words: List[str], embeddings: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Find the word that doesn't belong (lowest avg similarity to others)."""
    # Your solution here
    pass


def cluster_words(words: List[str], embeddings: Dict[str, np.ndarray], n_clusters: int = 3) -> Dict[int, List[str]]:
    """Cluster words using simple k-means."""
    # Your solution here
    pass


def reduce_dimensions(embeddings: Dict[str, np.ndarray], method: str = "pca", n_components: int = 2) -> Dict[str, np.ndarray]:
    """Reduce embedding dimensions using PCA."""
    # Your solution here
    pass


def visualize_embeddings(words: List[str], embeddings: Dict[str, np.ndarray], labels: Dict[str, str] = None, title: str = "Word Embeddings") -> Dict[str, Tuple[float, float]]:
    """Get 2D coordinates for visualization."""
    # Your solution here
    pass


def word_arithmetic(positive: List[str], negative: List[str], embeddings: Dict[str, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]:
    """Perform word vector arithmetic: sum(positive) - sum(negative)."""
    # Your solution here
    pass


def embedding_statistics(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute statistics: vocab_size, embedding_dim, avg_norm, etc."""
    # Your solution here
    pass


def find_similar_pairs(embeddings: Dict[str, np.ndarray], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    """Find pairs of words with similarity above threshold."""
    # Your solution here
    pass


def semantic_field(seed_words: List[str], embeddings: Dict[str, np.ndarray], expansion_rounds: int = 2, top_k_per_word: int = 5, similarity_threshold: float = 0.6) -> Set[str]:
    """Expand seed words to find related semantic field."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    # Create mock embeddings with structure for meaningful tests
    def create_mock_embeddings():
        emb = {}
        emb["apple"] = np.array([1.0, 0.1, 0.0])
        emb["banana"] = np.array([0.9, 0.2, 0.0])
        emb["orange"] = np.array([0.95, 0.15, 0.05])
        emb["cat"] = np.array([0.0, 1.0, 0.1])
        emb["dog"] = np.array([0.1, 0.9, 0.2])
        emb["bird"] = np.array([0.05, 0.95, 0.1])
        emb["hammer"] = np.array([0.0, 0.0, 1.0])
        emb["saw"] = np.array([0.1, 0.1, 0.9])
        for word in emb:
            emb[word] = emb[word] / np.linalg.norm(emb[word])
        return emb
    
    embeddings = create_mock_embeddings()
    
    print("Testing find_nearest_neighbors...")
    
    # Test 1: Find neighbors
    neighbors = find_nearest_neighbors("apple", embeddings, top_k=3)
    assert len(neighbors) == 3, f"Test 1a failed: {len(neighbors)}"
    assert neighbors[0][0] in ["banana", "orange"], f"Test 1b failed: {neighbors[0]}"
    assert all(0 <= sim <= 1 for _, sim in neighbors), "Test 1c failed"
    print("  ✓ Nearest neighbors found")
    
    # Test 2: Exclude self
    neighbors = find_nearest_neighbors("apple", embeddings, top_k=3, exclude_self=True)
    assert "apple" not in [w for w, _ in neighbors], "Test 2 failed"
    print("  ✓ Self exclusion works")
    
    print("\nTesting analogy...")
    
    # Test 3: Basic analogy
    results = analogy("apple", "banana", "cat", embeddings, top_k=2)
    assert len(results) <= 2, f"Test 3a failed: {len(results)}"
    assert all(isinstance(r, tuple) for r in results), "Test 3b failed"
    print("  ✓ Analogy returns results")
    
    # Test 4: Exclude input words
    results = analogy("apple", "banana", "cat", embeddings, exclude_input=True)
    input_words = {"apple", "banana", "cat"}
    result_words = {w for w, _ in results}
    assert not input_words.intersection(result_words), f"Test 4 failed: {result_words}"
    print("  ✓ Input words excluded")
    
    print("\nTesting find_outlier...")
    
    # Test 5: Find outlier
    words = ["apple", "banana", "orange", "hammer"]
    outlier, sim = find_outlier(words, embeddings)
    assert outlier == "hammer", f"Test 5 failed: {outlier}"
    print("  ✓ Outlier found correctly")
    
    print("\nTesting cluster_words...")
    
    # Test 6: Clustering
    words = ["apple", "banana", "cat", "dog", "hammer", "saw"]
    clusters = cluster_words(words, embeddings, n_clusters=3)
    assert len(clusters) == 3, f"Test 6a failed: {len(clusters)}"
    all_clustered = []
    for words_in_cluster in clusters.values():
        all_clustered.extend(words_in_cluster)
    assert len(all_clustered) == 6, f"Test 6b failed: {len(all_clustered)}"
    print("  ✓ Clustering works")
    
    print("\nTesting reduce_dimensions...")
    
    # Test 7: PCA reduction
    reduced = reduce_dimensions(embeddings, method="pca", n_components=2)
    assert len(reduced) == len(embeddings), "Test 7a failed"
    for word, vec in reduced.items():
        assert vec.shape == (2,), f"Test 7b failed for {word}: {vec.shape}"
    print("  ✓ Dimension reduction works")
    
    print("\nTesting visualize_embeddings...")
    
    # Test 8: Visualization coordinates
    words = ["apple", "banana", "cat", "dog"]
    coords = visualize_embeddings(words, embeddings)
    assert len(coords) == 4, f"Test 8a failed: {len(coords)}"
    for word, (x, y) in coords.items():
        assert isinstance(x, (int, float)), f"Test 8b failed: {type(x)}"
        assert isinstance(y, (int, float)), f"Test 8c failed: {type(y)}"
    print("  ✓ Visualization coordinates computed")
    
    print("\nTesting word_arithmetic...")
    
    # Test 9: Word arithmetic
    results = word_arithmetic(positive=["apple", "cat"], negative=["banana"], embeddings=embeddings, top_k=3)
    assert len(results) <= 3, f"Test 9a failed: {len(results)}"
    assert all(isinstance(r, tuple) for r in results), "Test 9b failed"
    print("  ✓ Word arithmetic works")
    
    print("\nTesting embedding_statistics...")
    
    # Test 10: Statistics
    stats = embedding_statistics(embeddings)
    assert stats["vocab_size"] == 8, f"Test 10a failed: {stats['vocab_size']}"
    assert stats["embedding_dim"] == 3, f"Test 10b failed: {stats['embedding_dim']}"
    assert "avg_norm" in stats, "Test 10c failed"
    print("  ✓ Statistics computed")
    
    print("\nTesting find_similar_pairs...")
    
    # Test 11: Similar pairs
    pairs = find_similar_pairs(embeddings, threshold=0.9)
    assert isinstance(pairs, list), "Test 11a failed"
    for p in pairs:
        assert len(p) == 3, f"Test 11b failed: {p}"
        assert p[2] >= 0.9, f"Test 11c failed: similarity {p[2]}"
    print("  ✓ Similar pairs found")
    
    print("\nTesting semantic_field...")
    
    # Test 12: Semantic field expansion
    field = semantic_field(["apple"], embeddings, expansion_rounds=1, top_k_per_word=3, similarity_threshold=0.5)
    assert "apple" in field, "Test 12a failed"
    assert len(field) > 1, "Test 12b failed: should expand"
    assert "banana" in field or "orange" in field, f"Test 12c failed: {field}"
    print("  ✓ Semantic field expansion works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
